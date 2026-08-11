import {
  Alert,
  Button,
  Card,
  CardContent,
  Divider,
  Stack,
  Typography,
} from "@mui/material";
import { useContext, useState } from "react";
import { Link as RouterLink, useLocation, useNavigate } from "react-router-dom";
import { UserContext } from "../../utils/UserContext";
import EmailInputField from "../authentication/EmailInputField";
import PasswordInputField from "../authentication/PasswordInputField";

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

/**
 * A form that can be filled in with a user's account credentials to login
 * to the service.
 *
 * @returns {@mui.material.Card}
 */
const LoginForm = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const { refreshUser } = useContext(UserContext);
  const [loginError, setLoginError] = useState("");
  const [password, setPassword] = useState(null);
  const [isPasswordValid, setIsPasswordValid] = useState(false);
  const [alertPasswordRequired, setAlertPasswordRequired] = useState(false);
  const [email, setEmail] = useState(null);
  const [isEmailValid, setIsEmailValid] = useState(false);
  const [alertEmailRequired, setAlertEmailRequired] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  function validateEmail(e) {
    setAlertEmailRequired(false);
    setIsEmailValid(e.isValid);
    setEmail(e.email.trim());
  }

  function validatePassword(e) {
    setAlertPasswordRequired(false);
    setIsPasswordValid(e.isValid);
    setPassword(e.password);
  }

  function generateLoginAlert() {
    if (loginError) {
      return (
        <Alert variant="filled" severity="error">
          {loginError}
        </Alert>
      );
    }

    return null;
  }

  async function handleLogin(e) {
    e.preventDefault();

    setLoginError("");

    // Check if all input fields are valid.
    if (!isEmailValid) {
      setAlertEmailRequired(email === null);
      return;
    }
    if (!isPasswordValid) {
      setAlertPasswordRequired(password === null);
      return;
    }

    setIsLoading(true);

    // Post the fetch request with the supplied credentials.
    await fetch(`${API_BASE}/login`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        email: email,
        password: password,
      }),
      credentials: "include",
    })
      .then(async (response) => {
        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || "Login details are incorrect");
        }

        return data;
     })
    .then(async (data) => {
      await refreshUser();

      const from = location.state?.from || "/landing";

      navigate(from, { replace: true });
    })
    .catch((error) => {
      setLoginError(error.message);
    });

    setIsLoading(false);
  }

  return (
    <Card
      component="form"
      onSubmit={handleLogin}
      sx={{
        width: {
          xs: "auto",
          sm: "400px",
          md: "400px",
          lg: "600px",
          xl: "600px",
        },
        minHeight: "auto",
        boxShadow: { xs: "none", sm: 16 },
      }}
    >
      <CardContent>
        <Stack spacing={2}>
          {generateLoginAlert()}
          <EmailInputField
            onChange={validateEmail}
            showRequired={alertEmailRequired}
          />
          <PasswordInputField
            onChange={validatePassword}
            truncate={true}
            restrictLength={false}
            showRequired={alertPasswordRequired}
            requireCharacters={false}
          />
          <Button
            loading={isLoading}
            type="submit"
            variant="contained"
            sx={{
              py: "1rem",
              fontSize: "1rem",
            }}
          >
            Login
          </Button>
          <Button
            component={RouterLink}
            to="/register"
            variant="outlined"
            sx={{
              py: "1rem",
              fontSize: "1rem",
            }}
          >
            Create Account
          </Button>
          <Divider variant="middle" aria-hidden="true" sx={{ py: "5px" }} />
          <Stack
            direction="row"
            spacing={{ xs: 1 }}
            alignItems="center"
            style={{ justifyContent: "center" }}
          >
            <Typography align="center" variant="subtle">
              Forgot your password?
            </Typography>
            <Button
              component={RouterLink}
              to="/forgot-password"
              variant="outlined"
              align="center"
            >
              Click Here
            </Button>
          </Stack>
        </Stack>
      </CardContent>
    </Card>
  );
};

export default LoginForm;
