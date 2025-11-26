#!/usr/bin/env python3
"""
Web UI for Audio Transcription with Speaker Diarization.
Built with Streamlit for easy localhost access.
"""

import streamlit as st
import tempfile
import os
import sys
from pathlib import Path
import subprocess
import json
import time

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()
PYTHON_EXE = sys.executable

# Page configuration
st.set_page_config(
    page_title="Audio Transcriber",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div {
        background-color: #1f77b4;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def run_command_with_output(cmd: list, description: str, progress_bar=None):
    """
    Run a command and capture output for display in UI.
    
    Args:
        cmd: Command as list of strings
        description: Description of what the command does
        progress_bar: Streamlit progress bar (optional)
    
    Returns:
        Tuple of (success: bool, output: str)
    """
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=str(SCRIPT_DIR)
        )
        output = result.stdout + result.stderr
        return True, output
    except subprocess.CalledProcessError as e:
        error_output = (e.stdout or "") + (e.stderr or "")
        if not error_output:
            error_output = f"Команда завершилась с кодом {e.returncode}"
        return False, error_output
    except Exception as e:
        return False, f"Ошибка выполнения команды: {str(e)}"


def transcribe_with_diarization_web(
    audio_path: str,
    language: str = "ru",
    max_speakers: int = 6,
    clean_audio: bool = True,
    whisper_model: str = "large-v3",
    progress_container=None,
    progress_bar=None,
    status_text=None,
    log_placeholder=None
):
    """
    Run complete transcription pipeline with speaker diarization for web UI.
    
    Args:
        audio_path: Path to input audio file
        language: Language code (ru, en)
        max_speakers: Maximum number of speakers
        clean_audio: Whether to clean audio first
        whisper_model: Whisper model size to use
        progress_container: Streamlit container for progress updates
    
    Returns:
        Dictionary with results and output file paths, or None on error
    """
    results = {
        "success": False,
        "output_files": {},
        "error": None
    }
    
    try:
        audio_path = Path(audio_path).resolve()
        
        if not audio_path.exists():
            results["error"] = f"Аудио файл не найден: {audio_path}"
            return results
        
        if not audio_path.is_file():
            results["error"] = f"Указанный путь не является файлом: {audio_path}"
            return results
    except Exception as e:
        results["error"] = f"Ошибка при проверке файла: {str(e)}"
        return results
    
    try:
        # Step 1: Clean audio (optional)
        if clean_audio:
            if progress_bar:
                progress_bar.progress(10)
            if status_text:
                status_text.text("Шаг 1/4: Очистка аудио (удаление тишины, нормализация)...")
            if progress_container:
                progress_container.info("🧹 Очистка аудио...")
            cleaned_path = audio_path.parent / f"{audio_path.stem}_cleaned.wav"
            
            success, output = run_command_with_output(
                [PYTHON_EXE, str(SCRIPT_DIR / "clean_audio.py"), str(audio_path)],
                "Audio cleaning"
            )
            
            if not success:
                if progress_container:
                    progress_container.warning("⚠️ Очистка аудио не удалась, используем оригинальный файл")
                working_audio = audio_path
            else:
                working_audio = cleaned_path
                results["output_files"]["cleaned_audio"] = str(cleaned_path)
        else:
            working_audio = audio_path
    
        # Step 2: Whisper transcription
        if progress_bar:
            progress_bar.progress(30)
        if status_text:
            status_text.text(f"Шаг 2/4: Транскрибация с Whisper (модель: {whisper_model})...")
        if progress_container:
            progress_container.info(f"🎤 Транскрибация с Whisper (модель: {whisper_model})...")
        if log_placeholder:
            log_placeholder.info("💡 Это может занять несколько минут, особенно при первой загрузке модели...")
        
        json_file = working_audio.parent / f"{working_audio.stem}.json"
        
        success, output = run_command_with_output(
            [
                PYTHON_EXE, str(SCRIPT_DIR / "transcribe_whisper.py"),
                str(working_audio),
                "--lang", language,
                "--model", whisper_model
            ],
            "Whisper transcription"
        )
        
        if not success:
            error_msg = output[:1000] if output else 'Неизвестная ошибка'
            # Check for common Whisper model errors
            if "model" in error_msg.lower() and ("not found" in error_msg.lower() or "download" in error_msg.lower()):
                results["error"] = (
                    f"Ошибка загрузки модели Whisper '{whisper_model}':\n\n"
                    f"Модель будет автоматически загружена при первом использовании.\n"
                    f"Возможные причины:\n"
                    f"• Нет подключения к интернету (требуется для первой загрузки)\n"
                    f"• Недостаточно места на диске (~1.5 GB для large-v3)\n"
                    f"• Проблемы с сетью\n\n"
                    f"Попробуйте:\n"
                    f"• Использовать меньшую модель (small, medium)\n"
                    f"• Проверить подключение к интернету\n"
                    f"• Освободить место на диске\n\n"
                    f"Детали ошибки:\n{error_msg}"
                )
            else:
                results["error"] = f"Ошибка транскрибации: {error_msg}"
            return results
        
        if not json_file.exists():
            results["error"] = f"Файл транскрипции не создан: {json_file}"
            return results
        
        results["output_files"]["json"] = str(json_file)
        
        # Step 3: Speaker diarization
        if progress_bar:
            progress_bar.progress(70)
        if status_text:
            status_text.text(f"Шаг 3/4: Определение спикеров (макс. {max_speakers})...")
        if progress_container:
            progress_container.info(f"👥 Определение спикеров (макс. {max_speakers})...")
        
        tagged_json = working_audio.parent / f"{working_audio.stem}_tagged.json"
        
        success, output = run_command_with_output(
            [
                PYTHON_EXE, str(SCRIPT_DIR / "diarize_nemo.py"),
                str(working_audio),
                str(json_file),
                "--max-speakers", str(max_speakers)
            ],
            "Speaker diarization"
        )
        
        if not success:
            results["error"] = f"Ошибка диаризации: {output[:500] if output else 'Неизвестная ошибка'}"
            return results
        
        if not tagged_json.exists():
            results["error"] = f"Файл диаризации не создан: {tagged_json}"
            return results
        
        results["output_files"]["tagged_json"] = str(tagged_json)
        
        # Step 4: Convert to readable formats
        if progress_bar:
            progress_bar.progress(90)
        if status_text:
            status_text.text("Шаг 4/4: Конвертация в читаемые форматы...")
        if progress_container:
            progress_container.info("📝 Конвертация в читаемые форматы...")
        
        success, output = run_command_with_output(
            [PYTHON_EXE, str(SCRIPT_DIR / "convert_to_readable.py"), str(tagged_json)],
            "Format conversion"
        )
        
        if success:
            if progress_bar:
                progress_bar.progress(100)
            if status_text:
                status_text.text("✅ Обработка завершена!")
            if log_placeholder:
                log_placeholder.empty()
            output_base = working_audio.stem
            txt_file = working_audio.parent / f"{output_base}_transcript.txt"
            md_file = working_audio.parent / f"{output_base}_transcript.md"
            detailed_md_file = working_audio.parent / f"{output_base}_detailed.md"
            
            results["output_files"]["txt"] = str(txt_file) if txt_file.exists() else None
            results["output_files"]["markdown"] = str(md_file) if md_file.exists() else None
            results["output_files"]["detailed_markdown"] = str(detailed_md_file) if detailed_md_file.exists() else None
            
            # Load transcript content for display
            if txt_file.exists():
                with open(txt_file, "r", encoding="utf-8") as f:
                    results["transcript_text"] = f.read()
            
            if tagged_json.exists():
                with open(tagged_json, "r", encoding="utf-8") as f:
                    results["transcript_data"] = json.load(f)
            
            results["success"] = True
    except Exception as e:
        results["error"] = f"Неожиданная ошибка: {str(e)}"
        import traceback
        results["traceback"] = traceback.format_exc()
    
    return results


def main():
    # Header
    st.markdown('<p class="main-header">🎙️ Audio Transcriber</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Транскрибация аудио с разделением по спикерам</p>', unsafe_allow_html=True)
    
    # Sidebar with settings
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        language = st.selectbox(
            "Язык аудио",
            options=["ru", "en"],
            index=0,
            help="Выберите язык аудио файла"
        )
        
        max_speakers = st.slider(
            "Максимальное количество спикеров",
            min_value=1,
            max_value=6,
            value=6,
            help="Максимальное количество спикеров для диаризации"
        )
        
        whisper_model = st.selectbox(
            "Модель Whisper",
            options=["tiny", "base", "small", "medium", "large", "large-v3"],
            index=5,
            help="Размер модели Whisper. Больше = лучше качество, но медленнее"
        )
        
        clean_audio = st.checkbox(
            "Очистка аудио",
            value=True,
            help="Предобработка аудио (удаление тишины, нормализация)"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Информация о моделях")
        st.info("""
        **tiny** - Быстро, низкое качество (~39 MB)  
        **base** - Быстро, базовое качество (~74 MB)  
        **small** - Средне, хорошее качество (~244 MB)  
        **medium** - Медленно, очень хорошее (~769 MB)  
        **large-v3** - Очень медленно, лучшее качество (~1.5 GB)
        
        ⚠️ **Примечание:** Модель автоматически загружается при первом использовании. 
        Убедитесь, что есть подключение к интернету.
        """)
    
    # Main content area
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Загрузите аудио файл",
        type=["wav", "mp3", "flac", "m4a", "ogg", "aac", "wma"],
        help="Поддерживаемые форматы: WAV, MP3, FLAC, M4A, OGG, AAC, WMA"
    )
    
    # Initialize session state for results
    if 'transcription_results' not in st.session_state:
        st.session_state.transcription_results = None
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None
    
    # Check if we have saved results to display
    if st.session_state.transcription_results and st.session_state.transcription_results.get("success"):
        st.success("✅ Результаты предыдущей обработки сохранены")
        
        # Display saved results
        st.markdown("---")
        st.header("📄 Результаты")
        
        # Transcript preview
        if "transcript_text" in st.session_state.transcription_results:
            st.subheader("Предпросмотр транскрипции")
            st.text_area(
                "Транскрипция",
                value=st.session_state.transcription_results["transcript_text"],
                height=300,
                disabled=True,
                label_visibility="collapsed"
            )
        
        # Download buttons
        st.subheader("📥 Скачать результаты")
        col1, col2, col3 = st.columns(3)
        
        results = st.session_state.transcription_results
        if results["output_files"].get("txt"):
            try:
                with open(results["output_files"]["txt"], "rb") as f:
                    col1.download_button(
                        "📄 TXT файл",
                        f.read(),
                        file_name=Path(results["output_files"]["txt"]).name,
                        mime="text/plain"
                    )
            except:
                col1.info("Файл недоступен")
        
        if results["output_files"].get("markdown"):
            try:
                with open(results["output_files"]["markdown"], "rb") as f:
                    col2.download_button(
                        "📝 Markdown файл",
                        f.read(),
                        file_name=Path(results["output_files"]["markdown"]).name,
                        mime="text/markdown"
                    )
            except:
                col2.info("Файл недоступен")
        
        if results["output_files"].get("tagged_json"):
            try:
                with open(results["output_files"]["tagged_json"], "rb") as f:
                    col3.download_button(
                        "📊 JSON файл",
                        f.read(),
                        file_name=Path(results["output_files"]["tagged_json"]).name,
                        mime="application/json"
                    )
            except:
                col3.info("Файл недоступен")
        
        # Statistics
        if "transcript_data" in results:
            st.markdown("---")
            st.subheader("📊 Статистика")
            
            data = results["transcript_data"]
            if "segments" in data:
                segments = data["segments"]
                speakers = set(seg.get("speaker", "Unknown") for seg in segments)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Сегментов", len(segments))
                col2.metric("Спикеров", len(speakers))
                if segments:
                    duration = segments[-1].get("end", 0)
                    col3.metric("Длительность", f"{duration/60:.1f} мин")
        
        st.markdown("---")
        if st.button("🗑️ Очистить результаты", type="secondary"):
            st.session_state.transcription_results = None
            st.session_state.uploaded_file_name = None
            st.rerun()
        
        st.markdown("---")
    
    if uploaded_file is not None:
        # Display file info
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
        st.info(f"📁 Файл: {uploaded_file.name} ({file_size:.2f} MB)")
        
        # Process button
        if st.button("🚀 Начать транскрибацию", type="primary", use_container_width=True):
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Progress tracking with detailed status
                st.markdown("### 📊 Ход обработки")
                status_placeholder = st.empty()
                log_placeholder = st.empty()
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Run transcription with progress updates
                results = transcribe_with_diarization_web(
                    audio_path=tmp_path,
                    language=language,
                    max_speakers=max_speakers,
                    clean_audio=clean_audio,
                    whisper_model=whisper_model,
                    progress_container=status_placeholder,
                    progress_bar=progress_bar,
                    status_text=status_text,
                    log_placeholder=log_placeholder
                )
                
                if results and results["success"]:
                    # Save results to session state
                    st.session_state.transcription_results = results
                    st.session_state.uploaded_file_name = uploaded_file.name
                    
                    st.success("✅ Транскрибация завершена успешно!")
                    st.info("💾 Результаты сохранены и будут доступны даже после перезагрузки страницы")
                    
                    # Rerun to show saved results
                    st.rerun()
                
                elif results:
                    error_msg = results.get("error", "Неизвестная ошибка")
                    st.error(f"❌ Ошибка при обработке: {error_msg}")
                    if error_msg and len(error_msg) > 100:
                        with st.expander("🔍 Детали ошибки"):
                            st.code(error_msg, language="text")
                    if "traceback" in results:
                        with st.expander("📋 Полный traceback"):
                            st.code(results["traceback"], language="python")
                else:
                    st.error("❌ Не удалось начать обработку. Проверьте файл и попробуйте снова.")
            
            except Exception as e:
                st.error(f"❌ Произошла ошибка: {str(e)}")
                with st.expander("🔍 Детали ошибки"):
                    st.exception(e)
            
            finally:
                # Cleanup temporary file
                try:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                except:
                    pass
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Загрузите аудио файл выше, чтобы начать транскрибацию")
        
        with st.expander("ℹ️ Информация о приложении"):
            st.markdown("""
            ### Возможности:
            - 🎤 **Транскрибация** с помощью OpenAI Whisper
            - 👥 **Диаризация** спикеров с помощью NVIDIA NeMo
            - ⏱️ **Таймстампы** для каждого сегмента
            - 🌍 **Мультиязычность** (русский и английский)
            - 📄 **Экспорт** в TXT, Markdown и JSON
            
            ### Как использовать:
            1. Загрузите аудио файл
            2. Настройте параметры в боковой панели
            3. Нажмите "Начать транскрибацию"
            4. Дождитесь завершения обработки
            5. Скачайте результаты
            
            ### Примечание:
            Все обработка происходит локально на вашем компьютере.
            Данные не отправляются в облако.
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 1rem;'>"
        "Audio Transcriber with Speaker Diarization | Локальная обработка"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
