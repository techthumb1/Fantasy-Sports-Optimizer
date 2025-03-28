import axios from 'axios';

const BASE_URL = 'http://localhost:5000/api'; // Replace with your Flask API URL

export const getPlayerData = async () => {
    try {
        const response = await axios.get(`${BASE_URL}/players`);
        return response.data;
    } catch (error) {
        console.error("Error fetching player data:", error);
        throw error;
    }
};

export const getESPNData = async (endpoint: string) => {
    try {
        const response = await axios.get(`${BASE_URL}/espn/${endpoint}`);
        return response.data;
    } catch (error) {
        console.error("Error fetching ESPN data:", error);
        throw error;
    }
};

export const getSleeperData = async (endpoint: string) => {
    try {
        const response = await axios.get(`${BASE_URL}/sleeper/${endpoint}`);
        return response.data;
    } catch (error) {
        console.error("Error fetching Sleeper data:", error);
        throw error;
    }
};

export const getSportsDataIO = async (endpoint: string) => {
    try {
        const response = await axios.get(`${BASE_URL}/sportsdataio/${endpoint}`);
        return response.data;
    } catch (error) {
        console.error("Error fetching SportsDataIO data:", error);
        throw error;
    }
};
