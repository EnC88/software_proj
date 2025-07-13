# API Integration for Configuration Options

This document explains the integration of the `get_db_options()` and `get_osi_options()` functions with the Profile.tsx dropdowns.

## Overview

The Profile.tsx component has been updated to fetch configuration options from the backend API instead of using hardcoded arrays. The API endpoints now dynamically extract options from your actual data files, providing real-world options based on your system inventory.

## API Endpoints

Three new API endpoints have been added to `src/app.py` that extract options from your data files:

### 1. Operating Systems Options
- **Endpoint**: `GET /api/options/operating-systems`
- **Function**: `get_osi_options()`
- **Data Sources**: 
  - `data/processed/compatibility_analysis.json`
  - `data/processed/Webserver_OS_Mapping.csv`
  - `data/processed/Change_History.csv`
  - `data/raw/sor_hist.csv`
  - `data/raw/PCat.csv`
- **Response**: 
```json
{
  "operating_systems": [
    "Windows Server 2019",
    "Ubuntu 20.04 LTS",
    "Red Hat Enterprise Linux 8",
    // ... dynamically extracted from your data
  ]
}
```

### 2. Database Options
- **Endpoint**: `GET /api/options/databases`
- **Function**: `get_db_options()`
- **Data Sources**:
  - `data/processed/compatibility_analysis.json`
  - `data/processed/Webserver_OS_Mapping.csv`
  - `data/processed/Change_History.csv`
  - `data/raw/sor_hist.csv`
- **Response**:
```json
{
  "databases": [
    "MySQL 8.0",
    "PostgreSQL 14",
    "Oracle Database 19c",
    // ... dynamically extracted from your data
  ]
}
```

### 3. Web Server Options
- **Endpoint**: `GET /api/options/web-servers`
- **Function**: `get_web_server_options()`
- **Data Sources**:
  - `data/processed/compatibility_analysis.json`
  - `data/processed/Webserver_OS_Mapping.csv`
  - `data/raw/WebServer.csv`
  - `data/processed/Change_History.csv`
- **Response**:
```json
{
  "web_servers": [
    "Apache HTTP Server 2.4",
    "Nginx 1.22",
    "Microsoft IIS 10",
    // ... dynamically extracted from your data
  ]
}
```

## Data Extraction Logic

The API endpoints use sophisticated data extraction logic:

### 1. Primary Source: Compatibility Analysis
- Reads from `data/processed/compatibility_analysis.json`
- Extracts unique models and product types from server data
- Filters by relevant keywords (e.g., "APACHE", "WINDOWS", "MYSQL")

### 2. Secondary Sources: CSV Files
- Scans multiple CSV files for relevant columns
- Looks for columns containing keywords like "DATABASE", "OS", "SERVER"
- Extracts unique values and filters by relevance

### 3. Fallback Mechanism
- If no data is found in files, falls back to common options
- Ensures the dropdowns always have options available

## Frontend Integration

### Profile.tsx Updates

The Profile component has been updated with the following changes:

1. **State Management**: Added state for options and loading/error states
```typescript
const [operatingSystems, setOperatingSystems] = useState<string[]>([]);
const [databases, setDatabases] = useState<string[]>([]);
const [webServers, setWebServers] = useState<string[]>([]);
const [loading, setLoading] = useState(true);
const [error, setError] = useState<string | null>(null);
```

2. **API Fetching**: Added useEffect to fetch options on component mount
```typescript
useEffect(() => {
  const fetchOptions = async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch all options in parallel
      const [osResponse, dbResponse, wsResponse] = await Promise.all([
        fetch('/api/options/operating-systems'),
        fetch('/api/options/databases'),
        fetch('/api/options/web-servers')
      ]);

      // Process responses...
    } catch (err) {
      // Handle errors and fallback to hardcoded options
    } finally {
      setLoading(false);
    }
  };

  fetchOptions();
}, []);
```

3. **Loading State**: Added loading indicator while fetching options
4. **Error Handling**: Added error display and fallback to hardcoded options
5. **Fallback Mechanism**: If API fails, falls back to hardcoded options

### SystemConfiguration.tsx Updates

The SystemConfiguration component has been updated with similar changes for consistency.

## Benefits

1. **Real-World Data**: Options are extracted from your actual system inventory
2. **Dynamic Updates**: Options automatically update when your data changes
3. **Centralized Configuration**: Options are managed in one place (backend)
4. **Error Resilience**: Fallback to hardcoded options if API fails
5. **Loading States**: Better user experience with loading indicators
6. **Consistency**: Both Profile and SystemConfiguration components use the same approach

## Testing

Use the provided test script to verify the API endpoints:

```bash
python test_api_endpoints.py
```

This will test all three endpoints and verify they return the expected data structure.

## Usage

The integration is transparent to users. The dropdowns will:

1. Show a loading indicator while fetching options
2. Display the fetched options once loaded
3. Show an error message if the API fails
4. Fall back to hardcoded options if needed

## Data Sources Priority

The API endpoints check data sources in this order:

1. **Compatibility Analysis JSON** (`data/processed/compatibility_analysis.json`)
   - Most comprehensive source
   - Contains processed server information
   - Extracts models and product types

2. **Processed CSV Files**
   - `data/processed/Webserver_OS_Mapping.csv`
   - `data/processed/Change_History.csv`
   - Scans for relevant columns

3. **Raw Data Files**
   - `data/raw/WebServer.csv`
   - `data/raw/sor_hist.csv`
   - `data/raw/PCat.csv`

4. **Fallback Options**
   - Common hardcoded options if no data found

## Future Enhancements

1. **Database-Driven Options**: Options could be stored in a database table
2. **User-Specific Options**: Options could be filtered based on user permissions
3. **Caching**: Implement caching to reduce API calls
4. **Real-time Updates**: WebSocket integration for real-time option updates
5. **Advanced Filtering**: Filter options based on environment, manufacturer, etc. 