import React, { useState } from 'react';

interface SleeperData {
  id: string;
  username: string;
  // Add any other fields returned from the Sleeper API
}

const SleeperQuery: React.FC = () => {
  const [usernameOrId, setUsernameOrId] = useState<string>('');
  const [sleeperData, setSleeperData] = useState<SleeperData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleQuery = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`/api/sleeper/user/${usernameOrId}`);
      const data: SleeperData = await response.json();

      if (response.ok) {
        setSleeperData(data);
      } else {
        setError(data.error || 'Failed to fetch Sleeper user data');
      }
    } catch (err) {
      setError('Error fetching Sleeper user data');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>Query Sleeper User</h2>
      <input
        type="text"
        value={usernameOrId}
        onChange={(e) => setUsernameOrId(e.target.value)}
        placeholder="Enter Username or ID"
      />
      <button onClick={handleQuery}>Query Sleeper User</button>

      {loading && <p>Loading...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {sleeperData && (
        <div>
          <h3>Sleeper User Data</h3>
          <pre>{JSON.stringify(sleeperData, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default SleeperQuery;
