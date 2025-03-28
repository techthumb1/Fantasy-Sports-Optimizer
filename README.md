# 🏈 Fantasy Sports Optimizer

**AI-Powered Daily Player Performance Prediction System**

The Fantasy Sports Optimizer is a full-stack machine learning application designed to predict player performance using real-time data aggregated from three major sports APIs: **Sleeper**, **ESPN**, and **SportsDataIO**. It provides daily predictive insights, leveraging a robust backend pipeline and a modern frontend interface to support fantasy sports analysis, strategy, and automation.

---

## 🚀 Features

- 🔁 **Multi-API Integration**: Aggregates, normalizes, and stores data from Sleeper, ESPN, and SportsDataIO.
- ⚙️ **Daily Data Pipeline**: Automatically fetches and stores data with scheduled updates.
- 🧠 **Machine Learning Predictions**: Generates performance forecasts using statistical models and ML algorithms.
- 📊 **Real-Time Player Insights**: View projected stats, matchup data, betting props, and more.
- 🌐 **REST API**: Flask-based endpoints for querying player stats, schedules, and predictive data.
- 🖥️ **Modern Frontend**: Built with Next.js, Tailwind CSS, and TypeScript for responsive UI/UX.

---

## 📁 Backend Modules

- `backend/app.py`: Main Flask application entry point.
- `backend/api/`: API integrations for Sleeper, ESPN, and SportsDataIO.
- `backend/models/`: ML models, database schemas, and model training utilities.
- `backend/services/`: Business logic (e.g., player service, prediction service).
- `backend/utils/`: Data preprocessing, caching, data fetching, and integration helpers.
- `backend/routes.py`: Route definitions and blueprint setup.
- `backend/init_db.py`: Database initialization script.
- `backend/config.py`: Environment and API key configurations.

---

## 💻 Frontend Overview

- `frontend/src/`: Next.js app source – includes pages, components, and API services.
- `frontend/components/`: Reusable UI components for querying and displaying player data.
- `frontend/pages/`: Pages for dashboard, features, contact, and upload forms.
- `frontend/tailwind.config.ts`: Tailwind CSS configuration.
- `frontend/tsconfig.json`: TypeScript project configuration.

---

## 🛠️ Tech Stack

### Backend:
- **Python**, **Flask**, **SQLAlchemy**
- **APScheduler** – scheduled data jobs
- **XGBoost**, **scikit-learn** – model training
- **PostgreSQL / SQLite** – data storage

### Frontend:
- **Next.js (React + TypeScript)**
- **Tailwind CSS** – responsive, modern design
- **Axios** – API communication

---

## 🔌 API Endpoints

### Sleeper API
- `GET /api/sleeper/players/<sport>`
- `GET /api/sleeper/user/<username>`
- `GET /api/sleeper/league/<league_id>`

### ESPN API
- `GET /api/espn/scoreboard`
- `GET /api/espn/teams/<team>`
- `GET /api/espn/standings`

### SportsDataIO API
- `GET /api/sportsdataio/teams`
- `GET /api/sportsdataio/player-stats?season=2024&week=1`
- `GET /api/sportsdataio/schedules/<season>`

---

## 📈 Prediction Pipeline

1. **Data Ingestion** – Fetch player and game data from APIs.
2. **Normalization** – Clean and unify data formats.
3. **Feature Engineering** – Derive model features from stats and metadata.
4. **Model Prediction** – Apply ML model to generate daily performance forecasts.
5. **API Serving** – Serve predictions through Flask API.

---

## ✅ Getting Started

### 🔧 Backend Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
cd backend
python app.py
```

## Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## 📌 Roadmap

- [x] Data ingestion from APIs
- [x] Model training and predictions
- [x] REST API endpoints
- [x] Frontend query interface
- [ ] Historical analytics dashboard
- [ ] Docker containerization
- [ ] Full test coverage

## 📄 License

MIT License

## 🙌 Contributing

Contributions, feedback, and suggestions are welcome. Please open an issue or submit a pull request to get started.

## 🔗 Acknowledgments

- Sleeper API
- ESPN API (Unofficial)
- SportsDataIO

---

Designed for precision. Built for performance. Optimized for fantasy domination.

🏈 **Fantasy Sports Optimizer** is your ultimate tool for daily fantasy sports analysis and predictions.
