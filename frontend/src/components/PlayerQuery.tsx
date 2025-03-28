import { useEffect, useState } from 'react';

const PlayerQuery = () => {
  interface Player {
    id: number;
    name: string;
  }

  const [players, setPlayers] = useState<Player[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchPlayers = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/players'); // Replace with your actual API URL
        if (!response.ok) {
        setError('Failed to fetch players');
        }
        const data = await response.json();
        setPlayers(data);
      } catch (error) {
        if (error instanceof Error) {
          setError(error.message);
        } else {
          setError('An unknown error occurred');
        }
      }
    };

    fetchPlayers();
  }, []); // No need to include 'error' in the dependency array as 'setError' is stable

  if (error) {
    return <div>Error: {error}</div>;
  }

  return (
    <div>
      <h1>Player Data</h1>
      <ul>
        {players.map((player: Player) => (
          <li key={player.id}>{player.name}</li>
        ))}
      </ul>
    </div>
  );
};

export default PlayerQuery;
