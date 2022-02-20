import React, { FC, useState } from 'react';
import { makeStyles } from '@material-ui/core/styles';

import { getMessage } from '../utils/api';
import { isAuthenticated } from '../utils/auth';

const useStyles = makeStyles((theme) => ({
  link: {
    color: '#f3f6f4',
  },
}));


let ifDetected = false

export const Home: FC = () => {
  const [message, setMessage] = useState('');
  const [error, setError] = useState<string>('');
  const classes = useStyles();

  const queryBackend = async () => {
    try {
      const message = await getMessage();
      if (message['Payload'] == 'Detected') {
        ifDetected = true;
        setMessage(message['Location'] + '֍' + message['Payload'] + '֍' + message['Coordinates']);
      } else {
        setMessage(message['Location'] + '֍' + message['Payload']);
      }
    } catch (err) {
      setError(String(err));
    }
  };

  const msg = message?.split(" ")

  return (
    <>
      {!message && !error && (
        <a className={"link"} href="#" onClick={() => queryBackend()}>
          Poacher Search
        </a>
      )}
      {message && ifDetected && (
        // <p>
        <div className={"resultBox"}>
          {message.split("֍").map(msg => (<>{msg}<br/><br/></>))}
        {/* // </p> */}
        </div>
      )}
      {message && !ifDetected && (
        // <p>
        <div className={"resultBoxFalse"}>
          {message.split("֍").map(msg => (<>{msg}<br/><br/></>))}
        {/* // </p> */}
        </div>
      )}
      {error && (
        <p>
          Error: <code>{error}</code>
        </p>
      )}
      <a className={"link"} href="/admin">
        Admin Dashboard
      </a>
      <a className={"link"} href="/protected">
        Protected Route
      </a>
      {isAuthenticated() ? (
        <a className={"link"} href="/logout">
          Logout
        </a>
      ) : (
        <>
          <a className={"link"} href="/login">
            Login
          </a>
          <a className={"link"} href="/signup">
            Sign Up
          </a>
        </>
      )}
    </>
  );
};
