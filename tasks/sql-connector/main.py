from typing import Any, Union

import logging
import csv
import zipfile

from mysql.connector import CMySQLConnection
from psycopg2.extensions import connection

import mysql.connector
import psycopg2

from coretex import currentTaskRun, CustomDataset


def connectMysqlDatabase(connectionConfig: dict[str, str]) -> CMySQLConnection:
    logging.info(f">> [SQL Connector] Connecting with MySQL database \"{connectionConfig["database"]}\"...")

    try:
        conn = mysql.connector.connect(**connectionConfig)
    except mysql.connector.errors.Error as e:
        logging.error(f">> [SQL Connector] Error while connecting to database: {e}")

    return conn


def connectPostgresqlDatabase(connectionConfig: dict[str, str]) -> connection:
    logging.info(f">> [SQL Connector] Connecting with PostgreSQL database \"{connectionConfig["database"]}\"...")

    try:
        conn = psycopg2.connect(**connectionConfig)
    except psycopg2._psycopg.Error as e:
        logging.error(f">> [SQL Connector] Error while connecting to database: {e}")

    return conn


def fetchAllData(conn: Any, dataset: CustomDataset, queryGetTables: str, queryGetRows: str) -> None:
    cursor = conn.cursor()
    cursor.execute(queryGetTables)
    tables = cursor.fetchall()
    tables = [table[0] for table in tables]

    for table in tables:
        tableData = []

        cursor.execute(queryGetRows + f"'{table}'")
        columnNames = list(cursor.fetchall())
        columnNames = [name[0] for name in columnNames]

        cursor.execute(f"SELECT * FROM {table}")
        rows = list(cursor.fetchall())

        for row in rows:
            tableData.append(dict(zip(columnNames, list(row))))

        sampleNameCSV = f"{table}.csv"
        with open(sampleNameCSV, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames = columnNames)
            writer.writeheader()
            writer.writerows(tableData)

        sampleNameZIP = f"{table}.zip"
        with zipfile.ZipFile(sampleNameZIP, "w") as zipFile:
            zipFile.write(sampleNameCSV)

        dataset.add(sampleNameZIP)
        logging.info(f">> [SQL Connector] The sample \"{sampleNameZIP}\" has been added to the dataset \"{dataset.name}\"")

    cursor.close()
    conn.close()
    logging.info(">> [SQL Connector] Connection with database is closed")


def main() -> None:
    taskRun = currentTaskRun()
    databaseType = taskRun.parameters["database_type"]

    credentials = taskRun.parameters["credentials"]
    host = taskRun.parameters["host"]
    port = taskRun.parameters["port"]
    database = taskRun.parameters["database"]

    connectionConfig = {
        'user': credentials.username,
        'password': credentials.password,
        'host': host,
        'database': database,
        'port': port
    }

    if databaseType == "MySQL":
        conn: Union[CMySQLConnection, connection] = connectMysqlDatabase(connectionConfig)

        if conn.is_connected():
            dataset = CustomDataset.createDataset(f"{taskRun.id}-{database}", taskRun.projectId)
            queryGetTables = f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{database}'"
            queryGetRows = f"SELECT column_name FROM information_schema.columns WHERE table_schema = '{database}' AND table_name = "
            fetchAllData(conn, dataset, queryGetTables, queryGetRows)

        else:
            logging.warning(">> [SQL Connector] Problem with the database connection")

    if databaseType == "PostgreSQL":
        conn = connectPostgresqlDatabase(connectionConfig)

        if conn:
            dataset = CustomDataset.createDataset(f"{taskRun.id}-{database}", taskRun.projectId)
            queryGetTables = f"SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            queryGetRows = f"SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name = "
            fetchAllData(conn, dataset, queryGetTables, queryGetRows)

        else:
            logging.warning(">> [SQL Connector] Problem with the database connection")


if __name__ == "__main__":
    main()
