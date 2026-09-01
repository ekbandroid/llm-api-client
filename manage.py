"""Служебные команды: завести админа, сменить пароль, посмотреть пользователей.

    python manage.py create-admin <логин>
    python manage.py set-password <логин>
    python manage.py approve <логин>
    python manage.py list
"""

import getpass
import sys

import auth
import db


def _ask_password() -> str:
    """Спрашивает пароль дважды, не отображая ввод."""
    first = getpass.getpass("Пароль: ")
    if getpass.getpass("Повторите: ") != first:
        sys.exit("Пароли не совпадают.")
    if error := auth.validate_credentials("placeholder", first):
        sys.exit(error)
    return first


def create_admin(login: str) -> None:
    if error := auth.validate_credentials(login, "placeholder_ok"):
        sys.exit(error)
    if db.get_by_login(login):
        sys.exit(f"Логин {login} уже занят. Смените пароль: set-password {login}")
    user = db.create_user(login, auth.hash_password(_ask_password()), is_admin=True)
    print(f"Админ {user['login']} создан, статус {user['status']}.")


def set_password(login: str) -> None:
    user = db.get_by_login(login)
    if user is None:
        sys.exit(f"Пользователь {login} не найден.")
    db.set_password(user["id"], auth.hash_password(_ask_password()))
    print(f"Пароль для {user['login']} обновлён.")


def approve(login: str) -> None:
    user = db.get_by_login(login)
    if user is None:
        sys.exit(f"Пользователь {login} не найден.")
    db.set_status(user["id"], db.APPROVED)
    print(f"Доступ для {user['login']} открыт.")


def list_users() -> None:
    users = db.list_all()
    if not users:
        print("Пользователей нет.")
        return
    print(f"{'id':>3}  {'логин':<20} {'статус':<12} {'админ':<6} способ входа")
    for u in users:
        way = "пароль" if u["password_hash"] else ""
        way = " + ".join(filter(None, [way, "яндекс" if u["yandex_id"] else ""])) or "—"
        print(f"{u['id']:>3}  {u['login']:<20} {u['status']:<12} {'да' if u['is_admin'] else 'нет':<6} {way}")


COMMANDS = {
    "create-admin": (create_admin, True),
    "set-password": (set_password, True),
    "approve": (approve, True),
    "list": (list_users, False),
}


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] not in COMMANDS:
        sys.exit(__doc__)

    handler, needs_login = COMMANDS[args[0]]
    if needs_login and len(args) < 2:
        sys.exit(f"Укажите логин: python manage.py {args[0]} <логин>")

    db.init()
    handler(*args[1:2]) if needs_login else handler()


if __name__ == "__main__":
    main()
