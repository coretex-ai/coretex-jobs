from pathlib import Path

import logging
import csv
import zipfile

from mysql.connector import Error

import mysql.connector

from coretex import currentTaskRun, CustomDataset


def main() -> None:
    taskRun = currentTaskRun()
    user = taskRun.parameters["user"]
    password = taskRun.parameters["password"]
    host = taskRun.parameters["host"]
    port = taskRun.parameters["port"]
    database = taskRun.parameters["database"]

    configForConnection = {
        'user': user,
        'password': password,
        'host': host,
        #'unix_socket': '/Applications/MAMP/tmp/mysql/mysql.sock',
        'database': database,
        'port': port
    }
    try:
        logging.info("Connecting with database")
        conn = mysql.connector.connect(**configForConnection)
        if(conn.is_connected()):
            dataset = CustomDataset.createDataset(f"{taskRun.id}-{database}", taskRun.projectId)

            cursor = conn.cursor()
            cursor.execute(f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{database}'")
            tables = list(cursor.fetchall())
            tables = [table[0] for table in tables]
            
            for table in tables:
                dataFromTableForCSV: list[dict] = []
                
                cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_schema = '{database}' AND table_name = '{table}'")
                columnNames = list(cursor.fetchall())
                columnNames = [name[0] for name in columnNames]
                
                cursor.execute(f"SELECT * FROM {table}")
                rows = list(cursor.fetchall())
                
                for row in rows:
                    dataFromTableForCSV.append(dict(zip(columnNames, list(row))))

                with open(f"{table}.csv", "w", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=columnNames)
                    writer.writeheader()
                    writer.writerows(dataFromTableForCSV)
                
                with zipfile.ZipFile(f"{table}.zip", "w") as zipFile:
                    zipFile.write(f"{table}.csv")

                dataset.add(f"{table}.zip")

                Path(f"{table}.csv").unlink()
                Path(f"{table}.zip").unlink()

    except Error as e:
        logging.error(f"Error while connecting to database {e}")

    finally:
        if(conn.is_connected()):
            conn.close()
            logging.info("Connection with database is closed")
    

if __name__ == "__main__":
    main()
