/**
 * 提示词格式化工具
 * 提供提示词相关的格式化方法
 */

import { logger } from './logger.js';

class PromptFormatter {
    // 定义特殊分隔符
    static SEPARATORS = {
        START: '【T:', // 标签开始标记
        END: ':T】',   // 标签结束标记
        NEWLINE: '\n'  // 换行符用于分隔多个标签
    };

    // 定义标点符号映射表
    static PUNCTUATION_MAP = {
        // 基础标点
        '，': ',',
        '。': '.',
        '、': ',',
        '；': ';',
        '：': ':',
        '？': '?',
        '！': '!',

        // 引号
        '\u201C': '"', // 中文左双引号
        '\u201D': '"', // 中文右双引号
        '\u2018': "'", // 中文左单引号
        '\u2019': "'", // 中文右单引号

        // 括号
        '（': '(',
        '）': ')',
        '【': '[',
        '】': ']',
        '「': '{',
        '」': '}',
        '『': '{',
        '』': '}',
        '〔': '[',
        '〕': ']',
        '［': '[',
        '］': ']',
        '｛': '{',
        '｝': '}',
        '《': '<',
        '》': '>',
        '〈': '<',
        '〉': '>',

        // 其他符号
        '～': '~',
        '｜': '|',
        '·': '.',
        '…': '...',
        '━': '-',
        '—': '-',
        '──': '--',
        '－': '-',
        '＿': '_',
        '＋': '+',
        '×': '*',
        '÷': '/',
        '＝': '=',
        '＠': '@',
        '＃': '#',
        '＄': '$',
        '％': '%',
        '＾': '^',
        '＆': '&',
        '＊': '*',
    };

    // 定义需要检测的标点符号集合
    static PUNCTUATION_SET = new Set([',', '.', '，', '。']);

    /**
     * 在文本中查找最近的标点符号
     */
    static findNearestPunctuation(text, searchForward = true) {
        if (!text) return false;

        // 移除开头和结尾的空格
        const cleanText = searchForward ? text.trimStart() : text.trimEnd();
        if (!cleanText) return false;

        // 搜索第一个非空格字符
        const char = searchForward ? cleanText[0] : cleanText[cleanText.length - 1];
        return this.PUNCTUATION_SET.has(char);
    }

    /**
     * 格式化标签，生成四种格式
     */
    static formatTag(tagValue) {
        try {
            // 生成四种格式
            const format1 = ` ${tagValue}`;           // 空格+标签
            const format2 = ` ${tagValue},`;          // 空格+标签+逗号
            const format3 = `, ${tagValue}`;          // 逗号+空格+标签
            const format4 = `, ${tagValue},`;         // 逗号+空格+标签+逗号

            logger.debug(`标签格式化 | 结果:成功 | 原始值:${tagValue} | 格式1:"${format1}" | 格式2:"${format2}" | 格式3:"${format3}" | 格式4:"${format4}"`);

            return {
                format1,
                format2,
                format3,
                format4,
                insertedFormat: null // 实际插入的格式，初始为null
            };
        } catch (error) {
            logger.error(`标签格式化 | 结果:异常 | 错误:${error.message}`);
            return {
                format1: ` ${tagValue}`,
                format2: ` ${tagValue},`,
                format3: `, ${tagValue}`,
                format4: `, ${tagValue},`,
                insertedFormat: null
            };
        }
    }

    /**
     * 确定应该使用哪种格式
     */
    static determineFormatType(beforeText, afterText) {
        try {
            // 判断是否为空输入框
            if (!beforeText && !afterText) {
                // logger.debug('格式判断 | 结果:空输入框 | 使用格式:2');
                return 2; // 空输入框使用格式2（空格+标签+逗号）
            }

            // 检查前方文本是否只包含空格
            const hasTextBefore = beforeText.trim().length > 0;

            // 跨空格查找前后的标点符号
            const hasCommaBefore = this.findNearestPunctuation(beforeText, false);
            const hasCommaAfter = this.findNearestPunctuation(afterText, true);

            // 如果前方没有实际文本（只有空格），使用格式2
            if (!hasTextBefore) {
                // logger.debug('格式判断 | 结果:前方无文本 | 使用格式:2');
                return 2; // 使用格式2（空格+标签+逗号）
            }

            // 根据前后标点符号情况决定使用哪种格式
            let formatType;
            if (hasCommaBefore && hasCommaAfter) {
                formatType = 1; // 前后都有标点，使用格式1（空格+标签）
            } else if (hasCommaBefore && !hasCommaAfter) {
                formatType = 2; // 前有标点后无标点，使用格式2（空格+标签+逗号）
            } else if (!hasCommaBefore && hasCommaAfter) {
                formatType = 3; // 前无标点后有标点，使用格式3（逗号+空格+标签）
            } else {
                formatType = 4; // 前后都无标点，使用格式4（逗号+空格+标签+逗号）
            }

            // logger.debug(`格式判断 | 前标点:${hasCommaBefore} | 后标点:${hasCommaAfter} | 前方文本:${hasTextBefore} | 使用格式:${formatType}`);
            return formatType;

        } catch (error) {
            logger.error(`格式判断 | 结果:异常 | 错误:${error.message}`);
            return 2; // 发生错误时默认使用格式2
        }
    }

    /**
     * 格式化提示词用于API调用
     */
    static formatPromptForAPI(prompt) {
        // 暂时直接返回原始文本，不做格式化处理
        return {
            formattedText: prompt,
            extractedParts: [],
            originalText: prompt
        };
    }

    /**
     * 获取用于API调用的纯文本
     */
    static getAPIText(formatInfo) {
        // 直接返回原始文本
        return formatInfo?.formattedText || '';
    }

    /**
     * 将API返回的结果恢复为原始格式
     */
    static restorePromptFormat(apiResult, formatInfo) {
        // 直接返回API结果
        return apiResult;
    }

    /**
     * 格式化翻译后的文本
     * 根据用户设置的格式化选项进行处理
     * 注意：此方法负责处理所有翻译结果的格式（包括百度翻译和LLM翻译），
     * 后端不再进行任何格式预处理或后处理。
     */
    static formatTranslatedText(text) {
        try {
            if (!text) return '';

            // 记录原始文本用于日志
            const originalText = text;

            // 获取格式化选项（从全局 FEATURES 对象）
            const options = {
                punctuation: window.FEATURES?.translateFormatPunctuation ?? true,
                space: window.FEATURES?.translateFormatSpace ?? true,
                dots: window.FEATURES?.translateFormatDots ?? false,
                newline: window.FEATURES?.translateFormatNewline ?? false
            };

            let formattedText = text;

            // 根据是否保留换行符选择不同的处理方式
            if (options.newline) {
                // 保留换行符：按行处理
                const lines = text.split('\n');
                const formattedLines = lines.map(line => {
                    return this._formatLine(line, options);
                });
                formattedText = formattedLines.join('\n');
            } else {
                // 不保留换行符：整体处理（换行符会被空格替换逻辑处理）
                formattedText = this._formatLine(text, options);
            }

            // 记录日志
            if (originalText !== formattedText) {
                const enabledOptions = [];
                if (options.punctuation) enabledOptions.push('标点转换');
                if (options.space) enabledOptions.push('空格处理');
                if (options.dots) enabledOptions.push('点号处理');
                if (options.newline) enabledOptions.push('保留换行');

                const logFormatted = formattedText.length > 100 ?
                    formattedText.substring(0, 100) + '...' : formattedText;
                logger.debug(`文本格式化 | 选项:[${enabledOptions.join(', ')}] | 结果:"${logFormatted}"`);
            }

            return formattedText;

        } catch (error) {
            logger.error(`文本格式化 | 结果:异常 | 错误:${error.message}`);
            return text; // 发生错误时返回原始文本
        }
    }

    /**
     * 格式化单行文本
     * 根据选项执行对应的格式化操作
     */
    static _formatLine(line, options) {
        let formattedLine = line;

        // 1. 标点符号转换
        if (options.punctuation) {
            for (const [cnPunct, enPunct] of Object.entries(this.PUNCTUATION_MAP)) {
                formattedLine = formattedLine.split(cnPunct).join(enPunct);
            }
        }

        // 2. 处理连续点号
        if (options.dots) {
            formattedLine = formattedLine.replace(/\.{3,}/g, '...');
        }

        // 3. 处理多余空格
        if (options.space) {
            formattedLine = formattedLine
                .replace(/\s+/g, ' ')           // 多个空格转换为单个空格
                .replace(/\s*,\s*/g, ', ')      // 统一逗号后的空格
                .trim();                        // 去除首尾空格
        }

        return formattedLine;
    }

    /**
     * RU-форк: определение языка для перевода RU ↔ EN (с фоллбеком ZH).
     * Кириллица → ru, латиница → en, иероглифы → zh.
     */
    static detectLanguage(text) {
        try {
            if (!text) {
                return { from: 'en', to: 'ru' };
            }

            const hasCyrillic = /[а-яА-ЯёЁ]/.test(text);
            const hasChineseChars = /[\u4e00-\u9fff]/.test(text);
            const hasEnglishChars = /[a-zA-Z]/.test(text);

            let from, to, type;

            // Чистые случаи — приоритет RU/EN, ZH фоллбек
            if (hasCyrillic && !hasEnglishChars && !hasChineseChars) {
                from = 'ru'; to = 'en'; type = 'чистый русский';
            } else if (!hasCyrillic && hasEnglishChars && !hasChineseChars) {
                from = 'en'; to = 'ru'; type = 'чистый английский';
            } else if (!hasCyrillic && !hasEnglishChars && hasChineseChars) {
                from = 'zh'; to = 'ru'; type = 'чистый китайский';
            } else {
                // Смешанный — считаем единицы
                const ruChars = text.match(/[а-яА-ЯёЁ]/g) || [];
                const cnChars = text.match(/[\u4e00-\u9fff]/g) || [];
                const enWords = text.match(/[A-Za-z]+(?:['’\-][A-Za-z]+)*/g) || [];
                // Для RU-форка объединяем китайские и русские иероглифы как "не-английский" если надо, но отдельно считаем
                // Определяем доминирующий не-английский язык
                const ruUnits = ruChars.length;
                const cnUnits = cnChars.length;
                const enUnits = enWords.length;
                // Доминирующий локальный язык
                const localUnits = ruUnits + cnUnits;
                const hasRuDominant = ruUnits >= cnUnits;

                const rule = window.FEATURES?.mixedLangTranslateRule || 'auto_minor';

                // Интерпретируем старые правила to_zh как to_ru
                const effectiveRule = rule === 'to_zh' ? 'to_ru' : rule;

                switch (effectiveRule) {
                    case 'to_ru':
                        if (enUnits > 0 && localUnits === 0) { from = 'en'; to = 'ru'; type = 'смешанный→русский'; }
                        else if (localUnits > 0) { from = hasRuDominant ? 'ru' : 'zh'; to = hasRuDominant ? 'en' : 'ru'; type = hasRuDominant ? 'смешанный→английский' : 'смешанный→русский'; }
                        else { from = 'en'; to = 'ru'; type = 'смешанный'; }
                        break;
                    case 'to_en':
                        from = hasRuDominant ? 'ru' : (hasChineseChars ? 'zh' : 'ru');
                        to = 'en';
                        type = 'смешанный→английский';
                        break;
                    case 'auto_major':
                        if (localUnits > enUnits) {
                            from = hasRuDominant ? 'ru' : 'zh';
                            to = hasRuDominant ? 'en' : 'ru';
                            type = hasRuDominant ? 'смешанный-доминант RU→EN' : 'смешанный-доминант ZH→RU';
                        } else if (enUnits > localUnits) {
                            from = 'en'; to = 'ru'; type = 'смешанный-доминант EN→RU';
                        } else {
                            from = hasRuDominant ? 'ru' : 'en'; to = hasRuDominant ? 'en' : 'ru'; type = 'смешанный';
                        }
                        break;
                    case 'auto_minor':
                    default:
                        if (localUnits > enUnits) {
                            from = 'en'; to = 'ru'; type = 'смешанный-доминант RU/ZH, переводим EN→RU';
                        } else if (enUnits > localUnits) {
                            from = hasRuDominant ? 'ru' : 'zh'; to = hasRuDominant ? 'en' : 'ru'; type = hasRuDominant ? 'смешанный-доминант EN, переводим RU→EN' : 'смешанный-доминант EN, переводим ZH→RU';
                        } else {
                            from = hasRuDominant ? 'ru' : 'en'; to = hasRuDominant ? 'en' : 'ru'; type = 'смешанный';
                        }
                        break;
                }
            }

            logger.debug(`Определение языка | ${type} | ${from}→${to}`);
            return { from, to };

        } catch (error) {
            logger.error(`Определение языка | ошибка: ${error.message}`);
            return { from: 'en', to: 'ru' };
        }
    }

    static isMixedChineseEnglish(text) {
        if (!text) return false;
        const hasChinese = /[\u4e00-\u9fff]/.test(text);
        const hasCyrillic = /[а-яА-ЯёЁ]/.test(text);
        const hasEnglish = /[A-Za-z]/.test(text);
        return (hasChinese && hasEnglish) || (hasCyrillic && hasEnglish) || (hasChinese && hasCyrillic);
    }
    static isMixedRussianEnglish(text) {
        if (!text) return false;
        const hasCyrillic = /[а-яА-ЯёЁ]/.test(text);
        const hasEnglish = /[A-Za-z]/.test(text);
        return hasCyrillic && hasEnglish;
    }
}

export { PromptFormatter };