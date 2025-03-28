import React, { useEffect, useState } from 'react';
import { getESPNData } from '../services/api';

const ESPNQuery: React.FC = () => {
    const [espnData, setEspnData] = useState<any>(null);

    useEffect(() => {
        // Fetch ESPN data when the component is mounted
        const fetchData = async () => {
            try {
                const data = await getESPNData('nfl/scoreboard');
                setEspnData(data);
            } catch (error) {
                console.error("Error fetching ESPN data:", error);
            }
        };

        fetchData();
    }, []);

    return (
        <div>
            <h1>ESPN NFL Scoreboard</h1>
            {espnData ? <pre>{JSON.stringify(espnData, null, 2)}</pre> : <p>Loading...</p>}
        </div>
    );
};

export default ESPNQuery;
