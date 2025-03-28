import React, { useEffect, useState } from "react";
import Image from "next/image";

interface UserProfile {
  avatar: string;
  display_name: string;
  username: string;
  user_id: string;
  is_bot: boolean;
}

const UserProfile: React.FC = () => {
  const [userData, setUserData] = useState<UserProfile | null>(null);

  useEffect(() => {
    fetch("/api/sleeper/user/jrob77") // Replace with your backend API route
      .then((response) => response.json())
      .then((data) => setUserData(data))
      .catch((error) => console.error("Error fetching user data:", error));
  }, []);

  if (!userData) return <div>Loading...</div>;

  return (
    <div>
      <Image
        src={`https://sleepercdn.com/avatars/${userData.avatar}`}
        alt={`${userData.display_name}'s Avatar`}
        width={500} // Adjust width as needed
        height={500} // Adjust height as needed
      />
      <p>Username: {userData.username}</p>
      <p>User ID: {userData.user_id}</p>
      <p>{userData.is_bot ? "Bot Account" : "Human User"}</p>
    </div>
  );
};

export default UserProfile;
