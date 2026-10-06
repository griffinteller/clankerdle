import { useEffect, useState } from 'react'
import './App.css'

interface Datum {
  info: string,
  time: string,
}

const HOST_ENDPOINT: string = "https://clankerdle-host-app.fly.dev/info";

const addData = (info: string) => {
  fetch(`${HOST_ENDPOINT}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'text/plain'
    },
    body: info,
  });
};

const retrieveData = async (setData: React.Dispatch<React.SetStateAction<Datum[]>>) => {
  const response = await fetch(`${HOST_ENDPOINT}`);
  let data: Datum[] = await response.json();
  setData(data);
};

function App() {
  const [data, setData] = useState<Datum[]>([]);

  const [inputVal, setInputVal] = useState('');

  useEffect(() => {
    retrieveData(setData);
  }, []);

  return (
    <>
      <section id="center">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            addData(inputVal);
            setInputVal('');
            retrieveData(setData);
          }}
        >
          <input
            type="text"
            value={inputVal}
            onChange={(e) => setInputVal(e.target.value)}
          />
          <button type="submit">Add Data</button>
        </form>

        <button 
          onClick={() => retrieveData(setData)}
        >
          Retrieve Data
        </button>

        {/* Display  data with some outlines and dividers */}
        {data.map((datum, index) => (
          <div key={index} style={{ border: "1px solid grey", margin: "5px", padding: "5px" }}>
            <p>{datum.info}</p>
            <p>{datum.time}</p>
          </div>
        ))}
      </section>
    </>
  )
}

export default App
