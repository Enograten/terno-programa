from pathlib import Path
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parent / "source.zip"))
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import io
import json
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from src.errors import AnalysisError
from src.thermal_reader import load_thermal, read_rgb
from src.analyzer import analyze
from src.storage import build_report, save_report, csv_bytes, json_bytes, load_history
from src.comparison_table import comparison_rows, comparison_csv
from src.visualization import png_bytes
from demo.create_demo import create_demo

ROOT = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((ROOT/"config.yaml").read_text(encoding="utf-8"))
# Public visitors never read or write a shared measurement history.
if "public_workspace" not in st.session_state:
    st.session_state["public_workspace"] = tempfile.TemporaryDirectory(prefix="thermal_session_")
WORK = Path(st.session_state["public_workspace"].name)
OUTPUT = WORK / "output"

def load_history(root):
    return list(st.session_state.get("public_reports", {}).values())

def save_report(root, analysis, report):
    digest = hashlib.sha256(json_bytes(report)).hexdigest()
    st.session_state.setdefault("public_reports", {})[digest] = json.loads(json_bytes(report))
    return "session"

st.set_page_config(page_title="Термограма", page_icon=":material/thermostat:", layout="wide")
st.markdown("<style>"+zipfile.ZipFile(ROOT/"source.zip").read("assets/interface.css").decode("utf-8")+"</style>", unsafe_allow_html=True)


def read_upload(data, filename, byte_order, overrides_json):
    suffix = Path(filename).suffix.lower()
    with tempfile.TemporaryDirectory(dir=WORK, prefix="upload_") as folder:
        path = Path(folder)/("original"+suffix)
        path.write_bytes(data)
        thermal = load_thermal(path, byte_order=byte_order, overrides=json.loads(overrides_json))
    thermal.filename = Path(filename).name
    thermal.metadata["original_sha256"] = hashlib.sha256(data).hexdigest()
    return thermal


def history_view():
    st.subheader("Таблиця порівняння")
    history = load_history(OUTPUT)
    include_demo = False
    if any(r["source"] == "demo" for r in history):
        include_demo = st.checkbox("Показувати демонстраційні записи", value=False, key="history_demo")
    history = [r for r in history if include_demo or r["source"] != "demo"]
    if not history:
        st.info("Заповніть «Дані для збереження» та збережіть результат. Кожна область з'явиться тут окремим рядком.")
        return
    patients = sorted({r["patient_id"] for r in history})
    patient = st.selectbox("Учасник", patients + [None],
        format_func=lambda p: "Усі учасники" if p is None else p, key="history_patient")
    records = [r for r in history if patient is None or r["patient_id"] == patient]
    frame = pd.DataFrame(comparison_rows(records))
    a, b = st.columns(2)
    area = a.selectbox("Ділянка", [None]+sorted(frame["Ділянка"].unique()),
        format_func=lambda v: "Усі ділянки" if v is None else v, key="history_area")
    session = b.selectbox("Сеанс", [None]+sorted(frame["Сеанс"].unique()),
        format_func=lambda v: "Усі сеанси" if v is None else (v or "Не вказано"), key="history_session")
    if area is not None:
        frame = frame[frame["Ділянка"] == area]
    if session is not None:
        frame = frame[frame["Сеанс"] == session]
    st.caption("Один рядок — одна область одного збереженого аналізу. Повторні сеанси не об’єднуються. Порожні поля означають, що дані не вказано.")
    if frame.empty:
        st.info("За цими фільтрами немає вимірювань.")
        return
    st.dataframe(frame, hide_index=True, width="stretch", column_config={
        name: st.column_config.NumberColumn(format="%.3f" if name == "SD, °C" else "%.2f")
        for name in ["Середня, °C", "Мінімум, °C", "Максимум, °C", "SD, °C", "Діапазон, °C"]})
    st.download_button("Завантажити таблицю CSV", comparison_csv(frame.to_dict("records")),
        "comparison_table.csv", "text/csv", key="history_csv")
    st.caption(f"Рядків: {len(frame)}. CSV містить усі стовпці та лише відібрані записи; його можна відкрити в Excel.")
    if patient is None:
        st.caption("Для графіка оберіть одного учасника.")
        return
    with st.expander("Динаміка по датах"):
        st.caption("Графік показує останній збережений запис за день для кожної ділянки. У таблиці вище залишаються всі повтори та сеанси.")
        latest = frame.drop_duplicates(["Дата", "Ділянка"], keep="last")
        st.line_chart(latest.pivot(index="Дата", columns="Ділянка", values="Середня, °C"), y_label="°C")
        pairs = []
        for record in records:
            if session is not None and record.get("session", "") != session:
                continue
            comparison = record.get("comparison")
            if comparison and (area is None or area in [comparison["name_a"], comparison["name_b"]]):
                pairs.append({"Дата": record["measurement_date"],
                              "Пара": " / ".join(sorted([comparison["name_a"], comparison["name_b"]])),
                              "Асиметрія, °C": comparison["absolute_asymmetry_c"],
                              "Збережено": record.get("saved_at", record["created_at"])})
        if pairs:
            pf = pd.DataFrame(pairs).sort_values(["Дата","Збережено"]).drop_duplicates(["Дата","Пара"], keep="last")
            st.line_chart(pf.pivot(index="Дата", columns="Пара", values="Асиметрія, °C"), y_label="Δ °C")


def confirm_pending():
    pending = st.session_state.pop("pending_analysis", None)
    if pending is not None:
        st.session_state["analysis"] = pending.confirm_geometry()



def region_label(key):
    return key.replace("Region ", "Область ")


def settings():
    mode = st.radio("Джерело температур", ["FLIR", "Матриця °C", "Демонстрація"], key="mode")
    threshold = st.number_input("Температурний поріг, °C",
        value=float(CONFIG["default_threshold_c"]), min_value=-273.0,
        max_value=2000.0, step=.5, key="threshold",
        help="Використовується лише для показника «Частка вище порога».")
    with st.expander("Розпізнавання контуру"):
        annotation_style = st.selectbox("Тип обведення", ["auto", "line", "green_handles", "thin_line"],
            format_func=lambda value: {"auto": "Автоматично", "line": "Кольорова лінія",
                "green_handles": "Зелений контур із точками", "thin_line": "Тонка зелена / блакитна лінія"}[value], key="annotation_style")
        color = st.selectbox("Колір лінії", ["red", "green", "cyan", "magenta"],
            format_func=lambda value: {"red": "Червоний", "green": "Зелений",
                "cyan": "Блакитний", "magenta": "Пурпуровий"}[value], key="contour_color")
        gap = st.slider("Допустимий розрив, px", 0, 7, int(CONFIG["contour"]["max_gap_px"]), key="contour_gap")
        min_area = st.number_input("Мінімальна площа, пікселів сенсора", min_value=12,
            value=int(CONFIG["contour"]["min_area_sensor_pixels"]), step=10, key="contour_area")
    byte_order, overrides = "auto", {}
    if mode == "FLIR":
        with st.expander("Параметри зйомки FLIR"):
            st.caption("За замовчуванням використовуються параметри з файлу.")
            byte_order = st.selectbox("Порядок байтів RAW", ["auto", "native", "swap"], key="byte_order")
            if st.checkbox("Змінити параметри зйомки", value=False, key="custom_flir"):
                st.warning("Ці параметри змінюють розраховану температуру.")
                overrides["Emissivity"] = st.number_input("Коефіцієнт випромінювання", .01, 1., .95, .01)
                overrides["ObjectDistance"] = st.number_input("Відстань, м", 0., 10000., 1., .1)
                overrides["ReflectedApparentTemperature"] = st.number_input("Відбита температура, °C", -99., 999., 20.)
                overrides["AtmosphericTemperature"] = st.number_input("Температура повітря, °C", -99., 999., 20.)
                overrides["IRWindowTemperature"] = st.number_input("Температура IR-вікна, °C", -99., 999., 20.)
                overrides["IRWindowTransmission"] = st.number_input("Пропускання IR-вікна", .01, 1., 1., .01)
                overrides["RelativeHumidity"] = st.number_input("Вологість, %", 0., 100., 50.)/100
    return mode, threshold, annotation_style, color, gap, min_area, byte_order, overrides


def empty_stage(outlined=False):
    symbol = '<span class="contour-symbol" aria-hidden="true"></span>' if outlined else '<span class="sensor-symbol" aria-hidden="true">°C</span>'
    label = "Тут буде зображення з контуром" if outlined else "Тут буде оригінальний знімок"
    st.markdown(f'<div class="empty-stage">{symbol}<span class="stage-label">{label}</span></div>',
        unsafe_allow_html=True)


def show_distribution(temperatures, mask):
    values = temperatures[mask]
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-9:
        return
    counts, _ = np.histogram(values, bins=36)
    peak = int(counts.max())
    bars = []
    for index, count in enumerate(counts):
        height = float(count)/peak*48
        bars.append(f'<rect x="{index*10}" y="{50-height:.2f}" width="7" height="{height:.2f}" rx="1" fill="#bf7553"/>')
    st.markdown('<div class="distribution"><div class="distribution-title">Розподіл температури в області</div>'
        '<svg viewBox="0 0 360 52" preserveAspectRatio="none" role="img" aria-label="Гістограма температури">'
        + ''.join(bars) + '</svg><div class="distribution-scale">'
        + f'<span>{low:.2f} °C</span><span>{high:.2f} °C</span></div></div>', unsafe_allow_html=True)


with st.container(key="app_header"):
    title, options = st.columns([3, 1], vertical_alignment="center")
    with title:
        st.markdown('<div class="app-wordmark"><div class="eyebrow">АНАЛІЗ ТЕМПЕРАТУРИ</div>'
            '<h1>Термограма</h1><p>Температура всередині обведеної області</p></div>',
            unsafe_allow_html=True)
    with options:
        with st.popover("Налаштування", width="stretch"):
            mode, threshold, annotation_style, color, gap, min_area, byte_order, overrides = settings()

st.info("Веб-версія: знімки обробляються на сервері. Таблиця доступна лише у вашому поточному підключенні. Перед закриттям чи оновленням сторінки завантажте CSV; постійного архіву немає.")
analysis_tab, history_tab = st.tabs(["Аналіз", "Історія"])
with history_tab:
    history_view()

with analysis_tab:
    thermal, marked = None, None
    input_hash = mode
    with st.container(key="uploads"):
        if mode == "Демонстрація":
            st.info("Демонстрація · штучні дані, дві області.")
            thermal, marked = create_demo()
            left, right = st.columns(2, gap="medium")
            with left:
                with st.container(border=False, key="source_card"):
                    st.markdown('<div class="file-heading">Оригінал<span>01</span></div>', unsafe_allow_html=True)
                    st.image(thermal.reference, width="stretch")
            with right:
                with st.container(border=False, key="marked_card"):
                    st.markdown('<div class="file-heading">З обведенням<span>02</span></div>', unsafe_allow_html=True)
                    st.image(marked, width="stretch")
        else:
            left, right = st.columns(2, gap="medium")
            with left:
                with st.container(border=False, key="source_card"):
                    st.markdown('<div class="file-heading">Оригінал<span>01</span></div>', unsafe_allow_html=True)
                    st.caption("Файл із тепловізора FLIR" if mode == "FLIR" else "Таблиця температур у °C")
                    original = st.file_uploader(
                        "Оригінальна термограма" if mode == "FLIR" else "Температурна матриця у °C",
                        type=["jpg", "jpeg", "fff"] if mode == "FLIR" else ["npy", "npz", "csv", "tsv"],
                        key="original_"+mode, label_visibility="collapsed")
                    if original is not None:
                        data = original.getvalue()
                        input_hash += hashlib.sha256(data).hexdigest()
                        try:
                            with st.spinner("Читаємо температури…"):
                                thermal = read_upload(data, original.name, byte_order, json.dumps(overrides, sort_keys=True))
                            preview = thermal.original_preview if thermal.original_preview is not None else thermal.reference
                            st.image(preview, width="stretch", alt="Оригінальна термограма")
                        except AnalysisError as exc:
                            st.error(str(exc))
                    else:
                        empty_stage()
            with right:
                with st.container(border=False, key="marked_card"):
                    st.markdown('<div class="file-heading">З обведенням<span>02</span></div>', unsafe_allow_html=True)
                    st.caption("Той самий кадр із готовим контуром")
                    marked_upload = st.file_uploader("Зображення з обведеною областю",
                        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
                        key="marked_"+mode, label_visibility="collapsed")
                    if marked_upload is not None:
                        marked_data = marked_upload.getvalue()
                        input_hash += hashlib.sha256(marked_data).hexdigest()
                        try:
                            marked = read_rgb(marked_data)
                            st.image(marked, width="stretch", alt="Термограма з готовим контуром")
                        except AnalysisError as exc:
                            st.error(str(exc))
                    else:
                        empty_stage(outlined=True)

    signature = hashlib.sha256((input_hash+json.dumps(
        [color, gap, min_area, byte_order, overrides, annotation_style], sort_keys=True)).encode()).hexdigest()
    if st.session_state.get("input_signature") != signature:
        for state_key in ("pending_analysis", "analysis", "saved_notice", "saved_report_signature"):
            st.session_state.pop(state_key, None)
        st.session_state["input_signature"] = signature

    with st.container(key="actions"):
        action, hint = st.columns([1, 2], vertical_alignment="center")
        with action:
            run = st.button("Аналізувати", type="primary", width="stretch",
                disabled=thermal is None or marked is None, key="analyze")
        with hint:
            if thermal is None or marked is None:
                st.caption("Додайте оригінал і копію з контуром.")
            else:
                st.caption("Зображення завантажено.")
    if run:
        for state_key in ("pending_analysis", "analysis", "saved_notice", "saved_report_signature"):
            st.session_state.pop(state_key, None)
        try:
            with st.spinner("Знаходимо контур та перевіряємо зображення…"):
                candidate = analyze(thermal, marked, color, gap, int(min_area), annotation_style)
                state_key = "pending_analysis" if candidate.geometry.get("requires_confirmation") else "analysis"
                st.session_state[state_key] = candidate
        except AnalysisError as exc:
            st.error(str(exc))

    pending = st.session_state.get("pending_analysis")
    if pending:
        st.divider()
        st.subheader("Перевірте область")
        picture, detail = st.columns([1, 1], gap="large")
        with picture:
            with st.container(key="mask_preview"):
                st.image(pending.overlay(list(pending.regions)), width="stretch", alt="Маска області для перевірки")
        with detail:
            st.warning("Контур відновлено з готової копії. Перед розрахунком перевірте розташування маски.")
            st.write("Маска має відповідати вашому обведенню та не захоплювати зайві ділянки.")
            st.button("Маска правильна — розрахувати", type="primary",
                key="confirm_geometry", on_click=lambda: confirm_pending())
            with st.expander("Деталі зіставлення"):
                st.json(pending.geometry)

    result = st.session_state.get("analysis")
    if result:
        st.divider()
        st.subheader("Результат")
        selected = list(result.regions)
        if len(selected) > 1:
            selected = st.multiselect("Області для аналізу", selected, default=selected,
                format_func=region_label, key="regions_"+result.analysis_id)
        if selected:
            # Reserve the main result area before rendering the optional report fields.
            overview = st.container(key="result_overview")
            names = {}
            with st.expander("Дані для збереження"):
                a, b = st.columns(2)
                patient = a.text_input("Код учасника / серії",
                    value="DEMO" if mode == "Демонстрація" else "Patient 001", key="patient")
                date = b.date_input("Дата вимірювання",
                    value=datetime.now(ZoneInfo("Europe/Kyiv")).date(), key="measurement_date")
                session = st.text_input("Сеанс", key="measurement_session", placeholder="Наприклад: 1 — до навантаження")
                conditions = st.text_area("Умови вимірювання", key="measurement_conditions",
                    placeholder="Наприклад: у спокої, після 15 хв адаптації, температура кімнати 22 °C")
                st.caption("Сеанс і фактичні умови вводяться вручну. Параметри FLIR зберігаються у звіті окремо.")
                for key in selected:
                    names[key] = st.text_input("Назва · "+region_label(key),
                        value=region_label(key), key=result.analysis_id+"name"+key).strip() or region_label(key)
                st.caption("Однакові назви областей поєднують вимірювання в історії. Нумерація — зліва направо на зображенні.")
            pair = None
            if len(selected) >= 2:
                with st.expander("Порівняння областей", expanded=True):
                    if st.checkbox("Порівняти дві області", value=True, key="compare_"+result.analysis_id):
                        c1, c2 = st.columns(2)
                        one = c1.selectbox("Перша область", selected, format_func=lambda k: names[k],
                            key="pair_a_"+result.analysis_id)
                        two = c2.selectbox("Друга область", [k for k in selected if k != one],
                            format_func=lambda k: names[k], key="pair_b_"+result.analysis_id)
                        pair = [one, two]
                    comparison_area = st.container()
            try:
                report = build_report(result, selected, threshold, patient, date, names, pair,
                                      session=session, measurement_conditions=conditions)
                with overview:
                    picture, detail = st.columns([1, 1], gap="large")
                    with picture:
                        with st.container(key="result_mask"):
                            st.image(result.overlay(selected), width="stretch", alt="Область, за якою розраховані температури")
                        st.caption("Вимірюється виділена область. Лінія контуру не враховується.")
                    with detail:
                        for row in report["regions"]:
                            with st.container(key="reading_"+row["region_id"]):
                                st.markdown(f"**{row['region_name']}**")
                                st.metric("Середня температура", f"{row['mean_temperature']:.2f} °C")
                                with st.container(key="summary_"+row["region_id"]):
                                    cols = st.columns(3)
                                    for col, (label, value) in zip(cols, [
                                        ("Мінімум", row["min_temperature"]),
                                        ("Максимум", row["max_temperature"]),
                                        ("Медіана", row["median_temperature"]),
                                    ]):
                                        col.metric(label, f"{value:.2f} °C")
                                temperature_range = row["max_temperature"] - row["min_temperature"]
                                st.markdown(f"Температурний діапазон (Tmax − Tmin): **{temperature_range:.2f} °C**")
                                show_distribution(result.thermal.temperatures, result.regions[row["region_id"]])
                                st.caption(f"Площа: {row['pixel_count']:,} пікселів")
                if report["comparison"]:
                    with comparison_area:
                        comp = report["comparison"]
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Різниця A − B", f"{comp['difference_c']:+.2f} °C")
                        c2.metric("Абсолютна асиметрія", f"{comp['absolute_asymmetry_c']:.2f} °C")
                        c3.metric("Відносна різниця", f"{comp['percentage_difference_kelvin']:.3f}%")
                        st.caption("Відносна різниця — від середньої абсолютної температури у Кельвінах.")
                with st.expander("Докладні показники"):
                    for row in report["regions"]:
                        st.markdown(f"**{row['region_name']}**")
                        st.write(f"Стандартне відхилення: {row['std']:.3f} °C · Вище {threshold:g} °C: {row['hot_area_percentage']:.2f}%")
                        st.dataframe(pd.DataFrame({label: [row[key]] for label, key in
                            [("P5, °C", "p5"), ("P25, °C", "p25"), ("P75, °C", "p75"), ("P95, °C", "p95")]}).round(3),
                            hide_index=True, width="stretch")
                    for warning in result.warnings:
                        st.warning(warning)
                    st.caption("Температури обчислено з даних сенсора. Точність залежить від тепловізора та параметрів зйомки.")
                    st.json(result.geometry, expanded=False)
                    st.download_button("Завантажити JSON", json_bytes(report),
                        "temperature_analysis.json", "application/json")
                if result.geometry.get("geometry_confirmed"):
                    st.caption("Розташування маски підтверджено вручну.")
                report_signature = hashlib.sha256(json_bytes(report)).hexdigest()
                if st.session_state.get("saved_report_signature") != report_signature:
                    st.session_state.pop("saved_notice", None)
                c1, c2 = st.columns(2)
                if c1.button("Зберегти результат", type="primary", key="save", width="stretch"):
                    if not patient.strip():
                        st.error("Укажіть код учасника або серії в «Дані для збереження».")
                    else:
                        saved = save_report(OUTPUT, result, report)
                        st.session_state["saved_notice"] = str(saved)
                        st.session_state["saved_report_signature"] = report_signature
                        st.rerun()
                c2.download_button("Завантажити CSV", csv_bytes(report),
                    "temperature_analysis.csv", "text/csv", width="stretch")
                if st.session_state.get("saved_notice"):
                    st.success("Результат збережено. Він доступний у вкладці «Історія».")
            except (AnalysisError, OSError) as exc:
                st.error(str(exc))
        else:
            st.info("Виберіть хоча б одну область.")

    if thermal is not None:
        with st.expander("Додаткові файли та дані"):
            if mode == "Демонстрація":
                buff = io.BytesIO()
                np.save(buff, thermal.temperatures, allow_pickle=False)
                st.download_button("Температурна матриця NPY", buff.getvalue(), "temperature.npy")
                st.download_button("Оригінал PNG", png_bytes(thermal.reference), "original.png")
                st.download_button("Обведений PNG", png_bytes(marked), "marked.png")
            else:
                st.caption("Якщо обведення не розпізнається, обведіть цей еталон і завантажте його як друге зображення.")
                st.download_button("Завантажити еталон для обведення", png_bytes(thermal.reference),
                    "thermal_reference.png", "image/png")
            st.json(thermal.metadata.get("effective_parameters", thermal.metadata), expanded=False)

st.markdown('<div class="footer-note">Веб-версія · Завантажте результати перед завершенням роботи</div>',
    unsafe_allow_html=True)
