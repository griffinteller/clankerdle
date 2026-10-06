import { useState } from 'react'
import './App.css'

interface Datum {
  info: string,
  time: string,
}

const HOST_ENDPOINT: string = "https://clankerdle-host-app.fly.dev";

const addData = async (info: string, setData: React.Dispatch<React.SetStateAction<Datum[]>>) => {
  const response = await fetch(`${HOST_ENDPOINT}/?info=${info}`);
  const newData: Datum = await response.json();
  setData(prevData => [...prevData, newData]);
};

function App() {
  const [data, setData] = useState<Datum[]>([]);

  const [inputVal, setInputVal] = useState('');

  return (
    <>
      <section id="center">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            addData(inputVal, setData);
            setInputVal('');
          }}
        >
          <input
            type="text"
            value={inputVal}
            onChange={(e) => setInputVal(e.target.value)}
          />
          <button type="submit">Add Data</button>
        </form>

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
