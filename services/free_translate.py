"""
Бесплатные сервисы перевода с авто-переключением (fallback chain).
Поддерживает: Google (бесплатный gtx), MyMemory, LibreTranslate.
Авто-выбор первого доступного, без API ключа.
"""

import asyncio
import json
import random
import re
import time
import httpx
from typing import Optional, Dict, Any, List

from ..utils.common import (
    ProgressBar, WARN_PREFIX, ERROR_PREFIX, PREFIX,
    TASK_TRANSLATE, SOURCE_FRONTEND
)
from .core import HTTPClientPool


class FreeTranslateService:
    """Цепочка бесплатных переводчиков с авто-фоллбеком."""

    # Порядок провайдеров по умолчанию (можно менять)
    DEFAULT_CHAIN = ["google", "mymemory", "libre", "yandex"]

    @staticmethod
    def split_text_by_paragraphs(text, max_length=4000):
        """Делим длинный текст на чанки, сохраняя абзацы."""
        if not text:
            return []
        lines = text.split('\n')
        chunks = []
        current = ""
        for line in lines:
            if len(line) > max_length:
                if current:
                    chunks.append(current)
                    current = ""
                remaining = line
                while remaining:
                    chunks.append(remaining[:max_length])
                    remaining = remaining[max_length:]
            elif current and (len(current) + len(line) + 1 > max_length):
                chunks.append(current)
                current = line
            else:
                current = f"{current}\n{line}" if current else line
        if current:
            chunks.append(current)
        return chunks or [text]

    # ---------- Google бесплатный (translate.googleapis.com) ----------

    @staticmethod
    async def _translate_google(client: httpx.AsyncClient, text: str, from_lang: str, to_lang: str) -> str:
        """
        Google бесплатный gtx endpoint.
        Не требует API ключа. Использует client=gtx.
        """
        # Нормализация языков для Google
        lang_map = {
            'zh': 'zh-CN',
            'zh-TW': 'zh-TW',
            'auto': 'auto',
        }
        sl = lang_map.get(from_lang, from_lang)
        tl = lang_map.get(to_lang, to_lang)
        # Google понимает 'zh' как китайский, но лучше zh-CN
        if sl == 'zh':
            sl = 'zh-CN'
        if tl == 'zh':
            tl = 'zh-CN'

        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "gtx",
            "sl": sl,
            "tl": tl,
            "dt": "t",
            "q": text
        }
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        resp = await client.get(url, params=params, headers=headers, timeout=12.0)
        if resp.status_code != 200:
            raise Exception(f"Google HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            data = resp.json()
            # data[0] - массив сегментов, каждый [translated, original, ...]
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                parts = []
                for seg in data[0]:
                    if seg and isinstance(seg, list) and len(seg) > 0 and seg[0]:
                        parts.append(seg[0])
                return "".join(parts)
            raise Exception(f"Неожиданный формат ответа Google: {str(data)[:300]}")
        except json.JSONDecodeError as e:
            raise Exception(f"Google JSON parse error: {e}")

    # ---------- MyMemory (api.mymemory.translated.net) ----------

    @staticmethod
    async def _translate_mymemory(client: httpx.AsyncClient, text: str, from_lang: str, to_lang: str) -> str:
        """
        MyMemory бесплатный - до 5000 символов за запрос, без ключа.
        """
        # MyMemory использует другие коды: ru, en, zh-CN
        lang_map = {
            'zh': 'zh-CN',
            'zh-TW': 'zh-TW',
            'auto': 'en',  # fallback если auto - пробуем en
        }
        sl = lang_map.get(from_lang, from_lang)
        tl = lang_map.get(to_lang, to_lang)
        if sl == 'auto':
            sl = 'en'
        # Для китайского MyMemory использует zh-CN
        if sl == 'zh':
            sl = 'zh-CN'
        if tl == 'zh':
            tl = 'zh-CN'

        url = "https://api.mymemory.translated.net/get"
        params = {
            "q": text,
            "langpair": f"{sl}|{tl}"
        }
        resp = await client.get(url, params=params, timeout=12.0)
        if resp.status_code != 200:
            raise Exception(f"MyMemory HTTP {resp.status_code}")
        data = resp.json()
        if data.get("responseStatus") == 429:
            raise Exception("MyMemory лимит запросов")
        translated = data.get("responseData", {}).get("translatedText")
        if not translated:
            # Проверка matches
            matches = data.get("matches", [])
            if matches:
                for m in matches:
                    if m.get("translation"):
                        translated = m["translation"]
                        break
        if not translated:
            raise Exception(f"MyMemory пустой ответ: {str(data)[:300]}")
        # MyMemory иногда возвращает с html entities
        import html
        return html.unescape(translated)

    # ---------- LibreTranslate (публичный инстанс) ----------

    @staticmethod
    async def _translate_libre(client: httpx.AsyncClient, text: str, from_lang: str, to_lang: str) -> str:
        """
        LibreTranslate публичный инстанс.
        """
        # Libre использует коды en, ru, zh, es, fr, etc.
        lang_map = {
            'zh': 'zh',
            'zh-TW': 'zh',
            'auto': 'en',
        }
        sl = lang_map.get(from_lang, from_lang)
        tl = lang_map.get(to_lang, to_lang)
        if sl == 'auto':
            sl = 'auto'  # Libre поддерживает auto
        if tl == 'zh':
            tl = 'zh'

        # Пробуем несколько публичных инстансов
        endpoints = [
            "https://libretranslate.com/translate",
            "https://translate.terraprint.co/translate",
            "https://translate.argosopentech.com/translate",
        ]
        last_err = None
        for url in endpoints:
            try:
                payload = {
                    "q": text,
                    "source": sl if sl != 'auto' else 'auto',
                    "target": tl,
                    "format": "text"
                }
                resp = await client.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=12.0)
                if resp.status_code != 200:
                    last_err = f"Libre HTTP {resp.status_code} at {url}"
                    continue
                data = resp.json()
                translated = data.get("translatedText")
                if translated:
                    return translated
                last_err = f"Libre пустой ответ at {url}: {str(data)[:200]}"
            except Exception as e:
                last_err = f"Libre error at {url}: {e}"
                continue
        raise Exception(last_err or "Libre все инстансы недоступны")

    # ---------- Yandex (неофициальный, через translate.yandex.net) ----------
    # Оставлен как опция, но требует ключ -> пропускаем, используем Libre вместо него.
    # Если у пользователя есть ключ, можно расширить.

    @staticmethod
    async def _translate_yandex_free(client: httpx.AsyncClient, text: str, from_lang: str, to_lang: str) -> str:
        """
        Попытка бесплатного Yandex через translate.yandex.net (без ключа, id trick).
        Не стабильно, оставлен для совместимости.
        """
        # Используем public id как в старых версиях
        url = "https://translate.yandex.net/api/v1.5/tr.json/translate"
        # Yandex коды: ru, en, zh, etc.
        lang_map = {'zh': 'zh', 'zh-TW': 'zh', 'auto': 'en'}
        sl = lang_map.get(from_lang, from_lang)
        tl = lang_map.get(to_lang, to_lang)
        # Для Yandex направление вида en-ru
        lang_pair = f"{tl}" if sl == 'auto' else f"{sl}-{tl}"
        # Пробуем без ключа с sid
        params = {
            "key": "trnsl.1.1.20130922T110455Z.affb96707e9ff39e.8a10a32683ae9ba8f666c451a545a0b2b4e2a8bde",
            "text": text,
            "lang": lang_pair,
            "format": "plain"
        }
        # Этот ключ - публичный демо, часто не работает. Если не работает - будет exception и fallback.
        resp = await client.get(url, params=params, timeout=12.0)
        if resp.status_code != 200:
            raise Exception(f"Yandex HTTP {resp.status_code}")
        data = resp.json()
        if data.get("code") != 200:
            raise Exception(f"Yandex error: {data.get('message', str(data))}")
        texts = data.get("text", [])
        if not texts:
            raise Exception("Yandex пустой ответ")
        return "".join(texts)

    # ---------- Унифицированный вызов с фоллбеком ----------

    # Карта провайдеров (используется через getattr, не напрямую)
    PROVIDER_FUNC_NAMES = {
        "google": "_translate_google",
        "mymemory": "_translate_mymemory",
        "libre": "_translate_libre",
        "yandex": "_translate_yandex_free",
    }

    @staticmethod
    async def translate(
        text: str,
        from_lang: str = 'auto',
        to_lang: str = 'ru',
        request_id: Optional[str] = None,
        chain: Optional[List[str]] = None,
        cancel_event=None,
        task_type=None,
        source=None
    ) -> Dict[str, Any]:
        """
        Основной метод: пробует провайдеры по цепочке, авто-переключение.
        
        chain: список провайдеров в порядке приоритета, например ["google", "mymemory"]
               Если None - используется DEFAULT_CHAIN.
        
        Возвращает {"success": bool, "data": {"translated": str, ...}, "error": str}
        """
        import time
        chain = chain or FreeTranslateService.DEFAULT_CHAIN
        # Нормализуем языки: zh->zh, ru->ru, en->en, auto->auto
        # Yandex/Google поддерживают разные коды, но нормализацию делаем внутри каждого провайдера

        if not text or text.strip() == '':
            return {"success": False, "error": "Текст для перевода пуст"}

        request_id = request_id or f"free_trans_{int(time.time())}_{random.randint(1000, 9999)}"
        task_type = task_type or TASK_TRANSLATE

        # Логирование языков
        lang_names = {
            "auto": "Авто", "zh": "Китайский", "en": "Английский", "ru": "Русский",
            "ja": "Японский", "ko": "Корейский", "ar": "Арабский", "es": "Испанский",
            "fa": "Персидский", "fr": "Французский", "pt-BR": "Португальский", "tr": "Турецкий",
            "zh-TW": "Трад. китайский"
        }
        from_name = lang_names.get(from_lang, from_lang)
        to_name = lang_names.get(to_lang, to_lang)

        # Разбиваем длинный текст
        chunks = FreeTranslateService.split_text_by_paragraphs(text, max_length=4000)
        if not chunks:
            chunks = [text]

        # Прогресс-бар
        from ..server import is_streaming_progress_enabled
        provider_display = f"FreeTranslate ({'→'.join(chain)})"
        pbar = ProgressBar(
            request_id=request_id,
            service_name=provider_display,
            streaming=is_streaming_progress_enabled(),
            extra_info=f"{from_name}→{to_name} | {len(text)} симв. | {len(chunks)} чанк(ов)",
            task_type=task_type,
            source=source
        )
        start_time = time.perf_counter()
        translated_parts = []

        # Создаем HTTP клиент (без прокси, trust_env авто)
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0, verify=False) as client:
            for idx, chunk in enumerate(chunks):
                # Проверка отмены
                if cancel_event is not None and cancel_event.is_set():
                    pbar.cancel(f"{WARN_PREFIX} Задача прервана | Сервис: FreeTranslate")
                    return {"success": False, "error": "Задача прервана", "interrupted": True}
                try:
                    from server import PromptServer
                    if hasattr(PromptServer.instance, 'execution_interrupted') and PromptServer.instance.execution_interrupted:
                        pbar.cancel(f"{WARN_PREFIX} Задача прервана | Сервис: FreeTranslate")
                        return {"success": False, "error": "Задача прервана", "interrupted": True}
                except:
                    pass

                # Пробуем каждый провайдер в цепочке
                last_error = None
                chunk_translated = None
                for provider in chain:
                    if provider not in FreeTranslateService.PROVIDER_FUNC_NAMES:
                        continue
                    try:
                        func = getattr(FreeTranslateService, f"_translate_{provider}")
                        result = await func(client, chunk, from_lang, to_lang)
                        if result and result.strip():
                            chunk_translated = result
                            # Логируем успех первого провайдера
                            if provider != chain[0]:
                                print(f"\r{WARN_PREFIX} FreeTranslate fallback сработал | Чанк {idx+1}/{len(chunks)} | Провайдер: {provider} (предыдущие не ответили)", flush=True)
                            break
                        else:
                            last_error = f"{provider}: пустой ответ"
                    except Exception as e:
                        last_error = f"{provider}: {str(e)[:200]}"
                        # Пробуем следующий
                        await asyncio.sleep(0.3)
                        continue

                if chunk_translated is None:
                    pbar.error(f"FreeTranslate все провайдеры недоступны для чанка {idx+1}: {last_error}")
                    return {"success": False, "error": f"Все сервисы перевода недоступны. Последняя ошибка: {last_error}"}

                translated_parts.append(chunk_translated)
                # Небольшая задержка между чанками чтобы не триггерить лимиты
                if idx < len(chunks) - 1:
                    await asyncio.sleep(0.5)

        translated_text = "\n".join(translated_parts)
        elapsed = int((time.perf_counter() - start_time) * 1000)
        pbar.done(char_count=len(translated_text), elapsed_ms=elapsed)

        return {
            "success": True,
            "data": {
                "translated": translated_text,
                "from": from_lang,
                "to": to_lang,
                "original": text,
                "provider_chain": chain
            }
        }

    @staticmethod
    async def translate_chunk(client, chunk, from_lang, to_lang, chain=None):
        """Упрощенный вызов для одного чанка (для совместимости)."""
        res = await FreeTranslateService.translate(chunk, from_lang, to_lang, chain=chain)
        if res.get("success"):
            return res["data"]["translated"]
        raise Exception(res.get("error", "Unknown error"))
