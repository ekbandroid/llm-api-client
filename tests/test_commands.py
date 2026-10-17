"""Юнит-тесты для commands.parse.

Поведение, под которое написан тест, взято из самого commands.py:

- строка без приставки «/» — не команда, parse отдаёт None (ход пойдёт
  обычным путём);
- известная команда с аргументом — parse отдаёт Условия, а аргумент
  становится question (для /help «какая ветка» так и есть);
- известная команда без аргумента — parse всё равно отдаёт Условия, но с
  готовым ответом (answer) и пустым question: к модели обращаться не нужно;
- неизвестная команда после слеша — в реестре её нет, parse отдаёт None.
"""

import unittest

import commands


class ParseОбычнаяСтрока(unittest.TestCase):
    """Строка без слеша — не команда."""

    def test_строка_без_слеша(self):
        self.assertIsNone(commands.parse("просто вопрос"))

    def test_пустая_строка(self):
        self.assertIsNone(commands.parse(""))


class ParseИзвестнаяКоманда(unittest.TestCase):
    """Известная команда разбирается в Условия."""

    def test_команда_с_аргументом(self):
        условия = commands.parse("/help какая ветка")
        self.assertIsNotNone(условия)
        self.assertEqual(условия.name, "help")
        self.assertEqual(условия.question, "какая ветка")

    def test_команда_без_аргумента(self):
        условия = commands.parse("/help")
        self.assertIsNotNone(условия)
        self.assertEqual(условия.name, "help")
        # Без аргумента ответ готов, к модели не идём.
        self.assertEqual(условия.question, "")
        self.assertEqual(условия.answer, commands.HELP_TEXT)


class ParseНеизвестнаяКоманда(unittest.TestCase):
    """Слеш есть, а команды с таким словом в реестре нет."""

    def test_неизвестная_команда(self):
        self.assertIsNone(commands.parse("/нетакой"))


if __name__ == "__main__":
    unittest.main()
