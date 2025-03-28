import React, { useState } from 'react';

interface SportsDataIOPlayer {
  id: string;
  name: string;
  // Add more fields expected from the SportsDataIO API
}

const SportsDataIOQuery: React.FC = () => {
  const [playerId, setPlayerId] = useState<string>('');
  const [sportsData, setSportsData] = useState<SportsDataIOPlayer | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleQuery = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`/api/sportsdataio/nfl/player/${playerId}`);
      const data: SportsDataIOPlayer = await response.json();

      if (response.ok) {
        setSportsData(data);
      } else {
        setError(data.error || 'Failed to fetch SportsDataIO player stats');
      }
    } catch (err) {
      setError('Error fetching player stats');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>Query SportsDataIO Player Stats</h2>
      <input
        type="text"
        value={playerId}
        onChange={(e) => setPlayerId(e.target.value)}
        placeholder="Enter Player ID"
      />
      <button onClick={handleQuery}>Query Player Stats</button>

      {loading && <p>Loading...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {sportsData && (
        <div>
          <h3>SportsDataIO Player Stats</h3>
          <pre>{JSON.stringify(sportsData, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default SportsDataIOQuery;
