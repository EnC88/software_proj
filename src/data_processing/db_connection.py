import os
import pandas as pd
import platform
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import logging
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import oracledb

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class DatabaseConnection:
    """Database connection manager for Oracle SOR_HISTORY database with proxy authentication."""
    
    def __init__(self):
        self.engine = None
        self.connection_string = None
        self._setup_connection()
    
    def _setup_connection(self):
        """Setup Oracle database connection with Kerberos authentication."""
        # Get Oracle database connection details from environment variables
        connection_name = os.getenv('DB_CONNECTION_NAME', 'eDART_DEV_DM')
        
        # For Kerberos authentication, we need to check for TNSNAMES.ora and krb5.conf
        logger.info(f"Setting up Kerberos connection with connection name: {connection_name}")
        
        # Check for Oracle environment variables
        oracle_home = os.getenv('ORACLE_HOME')
        tns_admin = os.getenv('TNS_ADMIN')
        
        if oracle_home:
            logger.info(f"ORACLE_HOME: {oracle_home}")
        if tns_admin:
            logger.info(f"TNS_ADMIN: {tns_admin}")
        
        # Check for krb5.conf
        krb5_conf = os.getenv('KRB5_CONFIG')
        if krb5_conf:
            logger.info(f"KRB5_CONFIG: {krb5_conf}")
        
        # For Kerberos, we typically use the connection name directly
        # The TNSNAMES.ora file contains the full connection details
        
        # Construct Oracle connection using Kerberos authentication
        try:
            # Initialize Oracle client
            self._init_oracle_client()
            
            # For Kerberos authentication, we use the connection name directly
            # The TNSNAMES.ora file contains all the connection details
            logger.info(f"Trying Kerberos connection with connection name: {connection_name}")
            
            # Approach 1: Try using connection name directly (TNSNAMES.ora)
            try:
                logger.info("Trying connection using TNSNAMES.ora...")
                connection_string = f"oracle+oracledb://{connection_name}"
                self.engine = create_engine(connection_string)
                
                # Test connection
                with self.engine.connect() as conn:
                    result = conn.execute(text("SELECT 1 FROM DUAL"))
                    logger.info(f"Successfully connected to Oracle database using TNSNAMES: {result.fetchone()}")
                    
                    # Check current user
                    user_result = conn.execute(text("SELECT USER FROM DUAL"))
                    current_user = user_result.fetchone()[0]
                    logger.info(f"Connected as user: {current_user}")
                    
                    # If we're not connected as EDART_DM, try proxy authentication
                    if current_user != 'EDART_DM':
                        logger.info("Not connected as EDART_DM, trying proxy authentication...")
                        return  # Let it try the next approach
                    else:
                        logger.info("Successfully connected as EDART_DM")
                        return
                    
            except SQLAlchemyError as e:
                logger.warning(f"TNSNAMES connection failed: {e}")
            
            # Approach 2: Try with proxy authentication
            try:
                logger.info("Trying connection with proxy authentication...")
                # Connect as I823577 and proxy to EDART_DM
                connection_string = f"oracle+oracledb://I823577@{connection_name}"
                self.engine = create_engine(
                    connection_string,
                    connect_args={
                        "proxy_user": "EDART_DM"
                    }
                )
                
                # Test connection
                with self.engine.connect() as conn:
                    result = conn.execute(text("SELECT 1 FROM DUAL"))
                    logger.info(f"Successfully connected to Oracle database with proxy: {result.fetchone()}")
                    
                    # Check current user
                    user_result = conn.execute(text("SELECT USER FROM DUAL"))
                    current_user = user_result.fetchone()[0]
                    logger.info(f"Connected as user: {current_user}")
                    
                    if current_user == 'EDART_DM':
                        logger.info("✅ Successfully connected as EDART_DM via proxy")
                        return
                    else:
                        logger.warning(f"Still connected as {current_user}, not EDART_DM")
                        # Try to switch to EDART_DM schema even if not connected as EDART_DM
                        try:
                            conn.execute(text("ALTER SESSION SET CURRENT_SCHEMA = EDART_DM"))
                            logger.info("Switched to EDART_DM schema")
                            return
                        except Exception as e:
                            logger.warning(f"Failed to switch to EDART_DM schema: {e}")
                    
            except SQLAlchemyError as e:
                logger.warning(f"Proxy connection failed: {e}")
            
            # Approach 3: Try direct oracledb connection with connection name
            try:
                logger.info("Trying direct oracledb connection with connection name...")
                direct_conn = oracledb.connect(dsn=connection_name)
                direct_conn.close()
                logger.info("Direct oracledb connection successful")
                
                # Create SQLAlchemy engine from direct connection
                connection_string = f"oracle+oracledb://{connection_name}"
                self.engine = create_engine(connection_string)
                
            except Exception as e:
                logger.warning(f"Direct oracledb connection failed: {e}")
            
            # If we get here, all approaches failed
            logger.error("All Kerberos connection approaches failed")
            logger.error("Please check:")
            logger.error("1. TNSNAMES.ora file is properly configured")
            logger.error("2. krb5.conf is set up correctly")
            logger.error("3. Kerberos ticket is valid (run 'klist' to check)")
            logger.error("4. Oracle client libraries are installed")
            self.engine = None
                
        except Exception as e:
            logger.error(f"Failed to connect to Oracle database: {e}")
            logger.error("Note: Check TNSNAMES.ora and Kerberos configuration")
            self.engine = None
    
    def get_table_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the Oracle database tables."""
        if not self.engine:
            return None
        
        try:
            with self.engine.connect() as conn:
                # Get all tables (not just user tables)
                tables_query = """
                SELECT owner, table_name 
                FROM all_tables 
                ORDER BY owner, table_name
                """
                tables = pd.read_sql(tables_query, conn)
                
                # Also get user tables
                user_tables_query = """
                SELECT table_name 
                FROM user_tables 
                ORDER BY table_name
                """
                user_tables = pd.read_sql(user_tables_query, conn)
                
                return {
                    'all_tables': tables.to_dict('records'),
                    'user_tables': user_tables['table_name'].tolist()
                }
                
        except SQLAlchemyError as e:
            logger.error(f"Error getting table info: {e}")
            return None
    
    def _init_oracle_client(self):
        """Initialize Oracle client library based on operating system."""
        try:
            if platform.system() == "Windows":
                # Windows Oracle client path
                oracledb.init_oracle_client(lib_dir=r"C:\Users\O803094\oracle\instantclient\instantclient_23_4")
            else:
                # macOS/Linux Oracle client path
                oracledb.init_oracle_client(lib_dir="../oracle")
            logger.info("Oracle client initialized successfully")
        except Exception as e:
            logger.warning(f"Oracle client initialization failed (this may be normal): {e}")
            # Continue without client initialization - it might work anyway
    
    def test_connection(self) -> bool:
        """Test if the database connection is working."""
        if not self.engine:
            return False
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1 FROM DUAL"))
                return True
        except SQLAlchemyError as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def switch_to_edart_dm_schema(self) -> bool:
        """Switch to EDART_DM schema."""
        if not self.engine:
            logger.error("No database connection available")
            return False
        
        try:
            with self.engine.connect() as conn:
                # Check current user
                current_user_query = "SELECT USER FROM DUAL"
                result = conn.execute(text(current_user_query))
                current_user = result.fetchone()[0]
                logger.info(f"Current user: {current_user}")
                
                # Check current schema
                schema_query = "SELECT SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') FROM DUAL"
                schema_result = conn.execute(text(schema_query))
                current_schema = schema_result.fetchone()[0]
                logger.info(f"Current schema: {current_schema}")
                
                # Switch to EDART_DM schema
                logger.info("Switching to EDART_DM schema...")
                alter_session_query = "ALTER SESSION SET CURRENT_SCHEMA = EDART_DM"
                conn.execute(text(alter_session_query))
                logger.info("Schema switched to EDART_DM")
                
                # Check schema after switch
                schema_after_result = conn.execute(text(schema_query))
                schema_after = schema_after_result.fetchone()[0]
                logger.info(f"Schema after switch: {schema_after}")
                
                # Check what tables we can see now using multiple approaches
                tables = []
                
                # Approach 1: Check user_tables (tables owned by current user)
                try:
                    user_tables_query = "SELECT table_name FROM user_tables ORDER BY table_name"
                    result = conn.execute(text(user_tables_query))
                    user_tables = [row[0] for row in result]
                    logger.info(f"User tables after schema switch ({len(user_tables)}): {user_tables}")
                    tables.extend(user_tables)
                except Exception as e:
                    logger.warning(f"Could not get user tables: {e}")
                
                # Approach 2: Check all_tables for EDART_DM schema
                try:
                    all_tables_query = """
                    SELECT table_name 
                    FROM all_tables 
                    WHERE owner = 'EDART_DM'
                    ORDER BY table_name
                    """
                    result = conn.execute(text(all_tables_query))
                    all_tables = [row[0] for row in result]
                    logger.info(f"EDART_DM tables from all_tables ({len(all_tables)}): {all_tables}")
                    tables.extend(all_tables)
                except Exception as e:
                    logger.warning(f"Could not get EDART_DM tables from all_tables: {e}")
                
                # Approach 2b: Check all_tables for any schema that might contain EDART_DM tables
                try:
                    all_tables_broad_query = """
                    SELECT owner, table_name 
                    FROM all_tables 
                    WHERE owner LIKE '%EDART%' OR owner LIKE '%DM%'
                    ORDER BY owner, table_name
                    """
                    result = conn.execute(text(all_tables_broad_query))
                    all_tables_broad = [(row[0], row[1]) for row in result]
                    logger.info(f"All tables in EDART/DM schemas ({len(all_tables_broad)}):")
                    for owner, table_name in all_tables_broad:
                        logger.info(f"  - {owner}.{table_name}")
                        if owner == 'EDART_DM':
                            tables.append(table_name)
                except Exception as e:
                    logger.warning(f"Could not get broad EDART tables: {e}")
                
                # Remove duplicates
                unique_tables = sorted(list(set(tables)))
                logger.info(f"Found {len(unique_tables)} unique tables in EDART_DM schema")
                
                if unique_tables:
                    logger.info(f"Tables in EDART_DM schema: {unique_tables[:10]}")  # Log first 10
                    return True
                else:
                    logger.warning("No tables found in EDART_DM schema")
                    return False
                    
        except SQLAlchemyError as e:
            logger.error(f"Error switching schema: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error switching schema: {e}")
            return False
    
    def list_edart_dm_tables(self) -> Optional[List[str]]:
        """List all tables available in EDART_DM schema."""
        if not self.engine:
            logger.error("No database connection available")
            return None
        
        try:
            with self.engine.connect() as conn:
                # First, check what user we're connected as
                user_result = conn.execute(text("SELECT USER FROM DUAL"))
                current_user = user_result.fetchone()[0]
                logger.info(f"Connected as user: {current_user}")
                
                # Check current schema
                schema_result = conn.execute(text("SELECT SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') FROM DUAL"))
                current_schema = schema_result.fetchone()[0]
                logger.info(f"Current schema: {current_schema}")
                
                # If we're not connected as EDART_DM, try to switch schema
                if current_user != 'EDART_DM':
                    logger.info("Not connected as EDART_DM, trying to switch schema...")
                    try:
                        alter_session_query = "ALTER SESSION SET CURRENT_SCHEMA = EDART_DM"
                        conn.execute(text(alter_session_query))
                        logger.info("Schema switched to EDART_DM")
                        
                        # Check schema after switch
                        schema_after_result = conn.execute(text("SELECT SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') FROM DUAL"))
                        schema_after = schema_after_result.fetchone()[0]
                        logger.info(f"Schema after switch: {schema_after}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to switch schema: {e}")
                        logger.info("Will try to access EDART_DM tables directly")
                
                # Try multiple approaches to get EDART_DM tables
                tables = []
                
                # Approach 1: Check user_tables (tables owned by current user)
                try:
                    user_tables_query = "SELECT table_name FROM user_tables ORDER BY table_name"
                    result = conn.execute(text(user_tables_query))
                    user_tables = [row[0] for row in result]
                    logger.info(f"User tables ({len(user_tables)}): {user_tables}")
                    tables.extend(user_tables)
                except Exception as e:
                    logger.warning(f"Could not get user tables: {e}")
                
                # Approach 2: Check all_tables for EDART_DM schema
                try:
                    all_tables_query = """
                    SELECT table_name 
                    FROM all_tables 
                    WHERE owner = 'EDART_DM'
                    ORDER BY table_name
                    """
                    result = conn.execute(text(all_tables_query))
                    all_tables = [row[0] for row in result]
                    logger.info(f"EDART_DM tables from all_tables ({len(all_tables)}): {all_tables}")
                    tables.extend(all_tables)
                except Exception as e:
                    logger.warning(f"Could not get EDART_DM tables from all_tables: {e}")
                
                # Approach 3: Check if we have privileges on EDART_DM tables
                try:
                    privileges_query = """
                    SELECT table_name 
                    FROM user_tab_privs 
                    WHERE table_schema = 'EDART_DM'
                    ORDER BY table_name
                    """
                    result = conn.execute(text(privileges_query))
                    privileged_tables = [row[0] for row in result]
                    logger.info(f"Tables with privileges ({len(privileged_tables)}): {privileged_tables}")
                    tables.extend(privileged_tables)
                except Exception as e:
                    logger.warning(f"Could not get privileged tables: {e}")
                
                # Remove duplicates and sort
                unique_tables = sorted(list(set(tables)))
                logger.info(f"Found {len(unique_tables)} unique tables in EDART_DM schema")
                
                return unique_tables
                    
        except SQLAlchemyError as e:
            logger.error(f"Error listing EDART_DM tables: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error listing EDART_DM tables: {e}")
            return None
    
    def get_sor_history_data(self, limit: int = 1000) -> Optional[List[Dict[str, Any]]]:
        """Get SOR_HISTORY data from EDART_DM.SOR_HISTORY table."""
        if not self.engine:
            logger.error("No database connection available")
            return None
        
        try:
            with self.engine.connect() as conn:
                # Check current user and schema
                user_result = conn.execute(text("SELECT USER FROM DUAL"))
                current_user = user_result.fetchone()[0]
                logger.info(f"Connected as user: {current_user}")
                
                # Check current schema
                schema_result = conn.execute(text("SELECT SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') FROM DUAL"))
                current_schema = schema_result.fetchone()[0]
                logger.info(f"Current schema: {current_schema}")
                
                # If not connected as EDART_DM, try to switch schema
                if current_user != 'EDART_DM':
                    logger.info("Not connected as EDART_DM, trying to switch schema...")
                    try:
                        conn.execute(text("ALTER SESSION SET CURRENT_SCHEMA = EDART_DM"))
                        logger.info("Switched to EDART_DM schema")
                    except Exception as e:
                        logger.warning(f"Failed to switch schema: {e}")
                        logger.info("Will try with full schema prefix")
                
                logger.info("Querying SOR_HISTORY table...")
                
                # Try multiple approaches to query the table
                query_attempts = [
                    # Approach 1: Full schema prefix
                    f"""
                    SELECT VERUMIDENTIFIER, REQUESTTRACKERID, OBJECTID, OBJECTNAME, 
                           RELATEDOBJECTID, RELATEDOBJECTNAME, ATTRIBUTENAME, OLDVALUE, 
                           NEWVALUE, VERUMSOR, VERUMCREATEDBY, VERUMCREATEDDATE, 
                           VERUMLASTMODIFIEDBY, VERUMMODIFIEDDATE, VERUMSTATUS, 
                           VERUMRETIREDDATE, ENTITLEMENTUSED
                    FROM EDART_DM.SOR_HISTORY
                    WHERE ROWNUM <= {limit}
                    ORDER BY VERUMCREATEDDATE DESC
                    """,
                    # Approach 2: Just table name (if schema switched)
                    f"""
                    SELECT VERUMIDENTIFIER, REQUESTTRACKERID, OBJECTID, OBJECTNAME, 
                           RELATEDOBJECTID, RELATEDOBJECTNAME, ATTRIBUTENAME, OLDVALUE, 
                           NEWVALUE, VERUMSOR, VERUMCREATEDBY, VERUMCREATEDDATE, 
                           VERUMLASTMODIFIEDBY, VERUMMODIFIEDDATE, VERUMSTATUS, 
                           VERUMRETIREDDATE, ENTITLEMENTUSED
                    FROM SOR_HISTORY
                    WHERE ROWNUM <= {limit}
                    ORDER BY VERUMCREATEDDATE DESC
                    """
                ]
                
                for i, query in enumerate(query_attempts, 1):
                    try:
                        logger.info(f"Trying query approach {i}...")
                        result = conn.execute(text(query))
                        rows = result.fetchall()
                        
                        # Convert to list of dictionaries
                        data = []
                        for row in rows:
                            row_dict = {
                                'VERUMIDENTIFIER': row[0],
                                'REQUESTTRACKERID': row[1],
                                'OBJECTID': row[2],
                                'OBJECTNAME': row[3],
                                'RELATEDOBJECTID': row[4],
                                'RELATEDOBJECTNAME': row[5],
                                'ATTRIBUTENAME': row[6],
                                'OLDVALUE': row[7],
                                'NEWVALUE': row[8],
                                'VERUMSOR': row[9],
                                'VERUMCREATEDBY': row[10],
                                'VERUMCREATEDDATE': row[11],
                                'VERUMLASTMODIFIEDBY': row[12],
                                'VERUMMODIFIEDDATE': row[13],
                                'VERUMSTATUS': row[14],
                                'VERUMRETIREDDATE': row[15],
                                'ENTITLEMENTUSED': row[16]
                            }
                            data.append(row_dict)
                        
                        logger.info(f"✅ Successfully retrieved {len(data)} rows from SOR_HISTORY using approach {i}")
                        return data
                        
                    except Exception as e:
                        logger.warning(f"Approach {i} failed: {e}")
                        continue
                
                logger.error("All query approaches failed")
                return None
                
        except SQLAlchemyError as e:
            logger.error(f"Error querying SOR_HISTORY: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error querying SOR_HISTORY: {e}")
            return None

# Global instance
db_connection = DatabaseConnection()

def main():
    """Test the database connection."""
    try:
        db_conn = DatabaseConnection()
        
        if db_conn.engine:
            print("✅ Database connection successful!")
            
            # Test basic connection
            if db_conn.test_connection():
                print("✅ Connection test passed!")
            else:
                print("❌ Connection test failed!")
                return
            
            # Test getting table info
            print("\n📋 Available tables:")
            table_info = db_conn.get_table_info()
            if table_info:
                print(f"Found {len(table_info.get('all_tables', []))} total tables")
                print(f"Found {len(table_info.get('user_tables', []))} user tables")
                
                # Show first 10 tables as examples
                all_tables = table_info.get('all_tables', [])
                if all_tables:
                    print("\nFirst 10 tables:")
                    for table in all_tables[:10]:
                        print(f"  - {table['owner']}.{table['table_name']}")
                
                user_tables = table_info.get('user_tables', [])
                if user_tables:
                    print(f"\nUser tables: {user_tables}")
            else:
                print("  No tables found or error occurred")
            
            # Test switching to EDART_DM schema
            print("\n🔄 Testing schema switch to EDART_DM...")
            if db_conn.switch_to_edart_dm_schema():
                print("✅ Successfully switched to EDART_DM schema!")
                
                # List all tables in EDART_DM schema
                print("\n📋 All tables in EDART_DM schema:")
                tables = db_conn.list_edart_dm_tables()
                if tables:
                    print(f"Found {len(tables)} tables:")
                    for table in tables:
                        print(f"  - {table}")
                else:
                    print("❌ No tables found or error occurred")
            else:
                print("❌ Failed to switch schema")
            
            # Test SOR_HISTORY data access
            print("\n🔍 Testing SOR_HISTORY data access...")
            sor_data = db_conn.get_sor_history_data(limit=5)
            if sor_data:
                print(f"✅ Successfully retrieved {len(sor_data)} rows from SOR_HISTORY!")
                print("\n📋 First 3 rows from SOR_HISTORY:")
                for i, row in enumerate(sor_data[:3], 1):
                    print(f"Row {i}:")
                    for key, value in row.items():
                        print(f"  {key}: {value}")
                    print("")
                
                # Show summary
                print("📊 Summary:")
                print(f"- Total rows retrieved: {len(sor_data)}")
                print(f"- Columns: {list(sor_data[0].keys()) if sor_data else 'None'}")
                
                # Check for specific object types
                object_names = set(row['OBJECTNAME'] for row in sor_data if row['OBJECTNAME'])
                print(f"- Object types found: {list(object_names)}")
            else:
                print("❌ Failed to retrieve SOR_HISTORY data")
            
        else:
            print("❌ Database connection failed!")
            
    except Exception as e:
        print(f"❌ Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 