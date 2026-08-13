"""Bootstrap public PostgreSQL schema: python -m framed.public_migrations."""

import os

from framed.public_store import PostgresPublicRepository


def main() -> None:
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        raise SystemExit("DATABASE_URL is required")
    PostgresPublicRepository(database_url).bootstrap()
    print("Public database migrations are current.")


if __name__ == "__main__":
    main()
