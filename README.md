# QR-Hint App

A web application for SQL query repair and analysis using the QR-Hint system.

## Project Structure

```
qr_hint_app/
├── backend/              # Flask backend API
│   ├── api/             # API routes
│   ├── services/        # Business logic (QR-Hint service)
│   ├── models/          # Data models
│   ├── qr_hint/         # QR-Hint core logic
│   ├── app.py           # Flask application entry point
│   └── config.py        # Configuration settings
│
└── frontend/            # React frontend
    ├── src/
    │   ├── components/  # React components
    │   ├── data/        # Questions data
    │   ├── services/    # API client
    │   └── App.jsx      # Main application component
    └── vite.config.js   # Vite configuration
```

## Features

- **Question Selection**: Select from predefined SQL questions (Q1-Q6)
- **Query Input**: Enter your SQL query for analysis
- **Query Repair**: Compare your query with the correct query and get repair suggestions
- **Visual Feedback**:
  - Red boxes show problems in your query
  - Green boxes show suggested fixes
  - Success message when your query is correct

## Setup and Installation

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the backend:
   ```bash
   python app.py
   ```

   The backend will run on `http://localhost:5000`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```

   The frontend will run on `http://localhost:5173`

## Usage

1. **Start both servers** (backend and frontend)

2. **Open your browser** and navigate to `http://localhost:5173`

3. **Select a question** from the sidebar (Q1-Q6)

4. **Enter your SQL query** in the text area
   - **IMPORTANT**: Do NOT add a semicolon (`;`) at the end of your query
   - The Calcite SQL parser used by QR-Hint doesn't support semicolons
   - Semicolons are automatically removed if you include them

5. **Click the "Repair" button** to analyze your query

6. **Review the results**:
   - If your query is correct, you'll see a success message
   - If there are issues, you'll see repair suggestions with:
     - Problem areas highlighted in red
     - Suggested fixes shown in green

## API Endpoints

### Backend API

- `GET /api/test-print` - Test endpoint to verify backend is running
- `POST /api/repair` - Analyze and repair SQL queries
  - Request body (note: no semicolons):
    ```json
    {
      "correct_query": "SELECT * FROM table",
      "incorrect_query": "SELECT * FORM table"
    }
    ```
  - Response:
    ```json
    {
      "ok": true,
      "repairs": [
        {
          "repair_site": "FORM",
          "fix": "FROM",
          "repair_site_size": 4,
          "fix_size": 4
        }
      ]
    }
    ```

## Technologies Used

### Backend
- Flask - Web framework
- Flask-CORS - Cross-origin resource sharing
- Z3 - SMT solver for query analysis
- Python 3.x

### Frontend
- React 19 - UI framework
- Vite - Build tool
- Tailwind CSS - Styling
- Fetch API - HTTP requests

## Configuration

### Backend Configuration (.env)
```
FLASK_ENV=development
HOST=0.0.0.0
PORT=5000
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

### Frontend Configuration (vite.config.js)
- Proxy configuration forwards `/api/*` requests to `http://localhost:5000`
- Development server runs on port 5173

## Troubleshooting

### "Query syntax error: Encountered ';' at line X, column Y"
- **Cause**: The Calcite SQL parser doesn't support semicolons at the end of queries
- **Solution**: Remove the semicolon from your query
- **Note**: The frontend automatically removes semicolons, but if you're testing the API directly, ensure queries don't end with `;`

### Backend not connecting
- Ensure the backend is running on port 5000
- Check CORS settings in `backend/config.py`

### Frontend API calls failing
- Verify the Vite proxy configuration in `frontend/vite.config.js`
- Check browser console for error messages
- Ensure both frontend and backend servers are running

### Query repair not working
- Check that the QR-Hint modules are properly installed
- Review backend logs for error messages
- Verify the query syntax is valid SQL (without semicolons)

## Development

### Adding New Questions
Edit `frontend/src/data/questions.js`:
```javascript
{
  id: 'q7',
  label: 'Q7',
  question: 'Your question description',
  correctQuery: 'SELECT * FROM table'  // No semicolon!
}
```

### Modifying API Endpoints
Backend routes are defined in `backend/api/routes.py`

### Styling Changes
The project uses Tailwind CSS. Modify component JSX files to update styles.

## License

[Your License Here]
