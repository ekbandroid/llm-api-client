"""Тесты на commands.parse.

Проверяют настоящее поведение разбора строки, а не желаемое: что считает
командой, когда возвращает None и как раскладывает аргумент.
"""

import unittest

import commands


class ParseTest(unittest.TestCase):
    def test_plain_string_is_not_a_command(self):
        """Строка без слеша — не команда, ход идёт обычным путём."""
        self.assertIsNone(commands.parse("просто вопрос без команды"))

    def test_known_command_with_argument(self):
        """Известная команда с аргументом отдаёт условия с вопросом."""
        условия = commands.parse("/help какая сейчас ветка")
        self.assertIsNotNone(условия)
        self.assertEqual(условия.name, "help")
        self.assertEqual(условия.question, "какая сейчас ветка")

    def test_known_command_without_argument(self):
        """Известная команда без аргумента — готовый ответ, а не вопрос к модели."""
        условия = commands.parse("/help")
        self.assertIsNotNone(условия)
        self.assertEqual(условия.name, "help")
        self.assertEqual(условия.question, "")
        self.assertEqual(условия.answer, commands.HELP_TEXT)

    def test_unknown_command(self):
        """Неизвестная команда — None, как и обычная строка."""
        self.assertIsNone(commands.parse("/нетакойкоманды аргумент"))


if __name__ == "__main__":
    unittest.main()
