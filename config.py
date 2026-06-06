# config.py
DB_CONFIG = {
    'host'    : 'localhost',
    'port'    : 5432,
    'database': 'retail_forecasting',  # new database for this project
    'user'    : 'postgres',
    'password': '4010983003pro'        # replace with your password
}

CONNECTION_STRING = (
    f"postgresql+psycopg2://"
    f"{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}"
    f"/{DB_CONFIG['database']}"
)