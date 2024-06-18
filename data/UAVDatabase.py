import sqlite3
from sqlite3 import Error

from data.UAVDataset import UAVDataset
from train.Hyperparams import Hyperparams
from data.Constants import DRONE_DB_PATH, RANGE_COL, COST_COL, VELOCITY_COL, SIM_RESULT_COL


class UAVDatabase(UAVDataset):
    def __init__(self, hparams: Hyperparams):
        super(UAVDataset, self).__init__()
        self.hparams = hparams
        self.connection = None
        self._initialize()
        
    def _initialize(self):
        self.connection = create_connection(DRONE_DB_PATH)
        assert self.connection is not None, "Error, could not connect to DB!"
        
    def add_design(self, config, metrics, result):
        query = f''' INSERT INTO drones(config, range, cost, velocity, result)
              VALUES('{config}',{float(metrics[RANGE_COL])},{float(metrics[COST_COL])},{float(metrics[VELOCITY_COL])},'{str(result)}') '''
        
        cur = self.connection.cursor()
        cur.execute(query)
        self.hparams.logger.log({"name": self.__name__, "msg": "Added design to DB"})
        
    def get_design(self, design_str):
        query = f"""SELECT * FROM drones WHERE config = {design_str}"""
        cur = self.connection.cursor()
        cur.execute(query)
        
        rows = cur.fetchall()
        
        if len(rows) == 0:
            return None
        else:
            return rows[0]

    def get_designs(self, design_str_list):
        query = f"""SELECT * FROM drones WHERE config IN ({','.join(design_str_list)})"""
        cur = self.connection.cursor()
        cur.execute(query)
    
        rows = cur.fetchall()
    
        if len(rows) == 0:
            return None
        else:
            return rows
        
    def close(self):
        self.connection.close()
    

"""
https://www.sqlitetutorial.net/sqlite-python/creating-tables/
"""


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return conn


def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)


def main():

    sql_drones_table = """ CREATE TABLE IF NOT EXISTS drones (
                                        config text PRIMARY KEY,
                                        range float,
                                        cost float,
                                        velocity float,
                                        result text
                                    ); """

    # create a database connection
    conn = create_connection(DRONE_DB_PATH)

    # create tables
    if conn is not None:
        # create projects table
        create_table(conn, sql_drones_table)
    else:
        print("Error! cannot create the database connection.")


if __name__ == '__main__':
    main()
