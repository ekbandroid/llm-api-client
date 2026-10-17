"""Юнит-тесты для commands.parse.

Проверяем настоящее поведение разбора строки, а не желаемое:
- строка без слеша — не команда, parse отдаёт None;
- известная команда с аргументом — аргумент уходит в question;
- неизвестная команда — None;
- известная команда без остатка — готовый ответ, а не запрос к модели.
"""

import unittest

import commands


class ParseNoSlashTest(unittest.TestCase):
    def test_plain_text_is_not_command(self):
        # Никакого слеша в начале — обычный ход, не команда.
        self.assertIsNone(commands.parse("какая сейчас ветка"))

    def test_empty_string_is_not_command(self):
        # Пустая строка — тоже не команда.
        self.assertIsNone(commands.parse(""))

    def test_slash_not_at_start_is_not_command(self):
        # Слеш не в начале строки команду не включает.
        self.assertIsNone(commands.parse("вопрос /help"))


class ParseKnownCommandWithArgumentTest(unittest.TestCase):
    def test_help_with_argument(self):
        условия = commands.parse("/help зачем нужен проект")
        self.assertIsNotNone(условия)
        self.assertEqual(условия.name, "help")
        # Аргумент (остаток после "/help ") уходит в question.
        self.assertEqual(условия.question, "зачем нужен проект")

    def test_argument_whitespace_is_stripped(self):
        # parse отрезает пробелы по краям остатка.
        условия = commands.parse("/help    где лежит проверка   ")
        self.assertEqual(условия.name, "help")
        self.assertEqual(условия.question, "где лежит проверка")


class ParseUnknownCommandTest(unittest.TestCase):
    def test_unknown_command_is_none(self):
        self.assertIsNone(commands.parse("/неизвестно что-то"))

    def test_unknown_command_with_argument_is_none(self):
        self.assertIsNone(commands.parse("/fly куда-нибудь"))


class ParseEmptyRemainderTest(unittest.TestCase):
    def test_help_without_argument_gives_answer(self):
        # "/help" без вопроса — готовый ответ, question пуст.
        условия = commands.parse("/help")
        self.assertIsNotNone(условия)
        self.assertEqual(условия.name, "help")
        self.assertEqual(условия.question, "")
        self.assertEqual(условия.answer, commands.HELP_TEXT)

    def test_help_with_trailing_space_gives_answer(self):
        # Пробел после команды — пустой остаток, тот же готовый ответ.
        условия = commands.parse("/help   ")
        self.assertEqual(условия.name, "help")
        self.assertEqual(условия.question, "")
        self.assertEqual(условия.answer, commands.HELP_TEXT)


if __name__ == "__main__":
    unittest.main()
