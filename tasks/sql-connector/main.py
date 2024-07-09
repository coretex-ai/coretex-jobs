from typing import Union

import logging
import csv
import zipfile

from mysql.connector import CMySQLConnection
from psycopg2.extensions import connection

import mysql.connector
import psycopg2

from coretex import currentTaskRun, CustomDataset


def connectMysqlDatabase(connectionConfig: dict[str, str]) -> CMySQLConnection:
    logging.info(f">> [SQL Connector] Connecting with MySQL database \"{connectionConfig['database']}\"...")

    try:
        conn = mysql.connector.connect(**connectionConfig)
    except mysql.connector.errors.Error as e:
        raise mysql.connector.errors.Error(f">> [SQL Connector] Error while connecting to database: {e}")

    return conn  # type: ignore[return-value]


def connectPostgresqlDatabase(connectionConfig: dict[str, str]) -> connection:
    logging.info(f">> [SQL Connector] Connecting with PostgreSQL database \"{connectionConfig['database']}\"...")

    try:
        conn: connection = psycopg2.connect(**connectionConfig)  # type: ignore[call-overload]
    except psycopg2._psycopg.Error as e:
        raise psycopg2._psycopg.Error(f">> [SQL Connector] Error while connecting to database: {e}")

    return conn


def fetchAllData(conn: Union[CMySQLConnection, connection], dataset: CustomDataset, queryGetTables: str, queryGetRows: str) -> None:
    cursor = conn.cursor()
    cursor.execute(queryGetTables)
    tables = cursor.fetchall()
    if len(tables) < 1:
        raise RuntimeError("There are no tables in the database")

    for table in [table[0] for table in tables]:
        tableData: list[dict[str, str]] = []

        cursor.execute(queryGetRows + f"'{table}'")  # type: ignore[str-bytes-safe]
        columnNames = list(cursor.fetchall())
        columnNames = [name[0] for name in columnNames]

        cursor.execute(f"SELECT * FROM {table}")  # type: ignore[str-bytes-safe]
        rows = list(cursor.fetchall())

        for row in rows:
            tableData.append(dict(zip(columnNames, list(row))))  # type: ignore[arg-type]

        sampleNameCsv = f"{table}.csv"  # type: ignore[str-bytes-safe]
        with open(sampleNameCsv, "w", newline = "") as file:
            writer = csv.DictWriter(file, fieldnames = columnNames)
            writer.writeheader()
            writer.writerows(tableData)  # type: ignore[arg-type]

        sampleNameZip = f"{table}.zip"  # type: ignore[str-bytes-safe]
        with zipfile.ZipFile(sampleNameZip, "w") as zipFile:
            zipFile.write(sampleNameCsv)

        dataset.add(sampleNameZip)
        logging.info(f">> [SQL Connector] The sample \"{sampleNameZip}\" has been added to the dataset \"{dataset.name}\"")

    cursor.close()
    conn.close()
    logging.info(">> [SQL Connector] Connection with database is closed")


def main() -> None:
    taskRun = currentTaskRun()
    databaseType = taskRun.parameters["databaseType"]
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
        mySqlConn = connectMysqlDatabase(connectionConfig)

        if mySqlConn.is_connected():
            dataset = CustomDataset.createDataset(f"{taskRun.id}-{database}", taskRun.projectId)
            queryGetTables = f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{database}'"
            queryGetRows = f"SELECT column_name FROM information_schema.columns WHERE table_schema = '{database}' AND table_name = "
            fetchAllData(mySqlConn, dataset, queryGetTables, queryGetRows)
        else:
            logging.warning(">> [SQL Connector] Problem with the database connection")
    elif databaseType == "PostgreSQL":
        postgreSqlConn = connectPostgresqlDatabase(connectionConfig)

        if postgreSqlConn:
            dataset = CustomDataset.createDataset(f"{taskRun.id}-{database}", taskRun.projectId)
            queryGetTables = f"SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            queryGetRows = f"SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name = "
            fetchAllData(postgreSqlConn, dataset, queryGetTables, queryGetRows)
        else:
            logging.warning(">> [SQL Connector] Problem with the database connection")


if __name__ == "__main__":
    main()
