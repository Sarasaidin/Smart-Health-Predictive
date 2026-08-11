import os
import smtplib
import traceback
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

def send_email(
    recipient: str,
    subject: str,
    content: str,
    content_type: str = "plainText"
):
    """
    Sends an email using Brevo SMTP.

    :param recipient: Recipient email address.
    :param subject: Email subject.
    :param content: Email body.
    :param content_type: 'plainText' or 'html'.
    :return: True if sent successfully, otherwise None.
    """

    try:
        smtp_host = os.getenv("SMTP_HOST", "smtp-relay.brevo.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_username = os.getenv("SMTP_USERNAME")
        smtp_password = os.getenv("SMTP_PASSWORD")
        sender_email = os.getenv("SMTP_FROM_EMAIL")
        sender_name = os.getenv(
            "SMTP_FROM_NAME",
            "Ever AI Technologies"
        )

        if not smtp_username:
            raise ValueError("SMTP_USERNAME is not configured")

        if not smtp_password:
            raise ValueError("SMTP_PASSWORD is not configured")

        if not sender_email:
            raise ValueError("SMTP_FROM_EMAIL is not configured")

        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = f"{sender_name} <{sender_email}>"
        message["To"] = recipient

        if content_type.lower() == "html":
            message.attach(MIMEText(content, "html"))
        else:
            message.attach(MIMEText(content, "plain"))

        print("===== BREVO EMAIL =====")
        print(f"Sending email to: {recipient}")
        print(f"Subject: {subject}")
        print(f"From: {sender_email}")

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.sendmail(
                sender_email,
                recipient,
                message.as_string()
            )

        print("Email sent successfully.")
        return True

    except Exception as ex:
        print("===== EMAIL ERROR =====")
        print(str(ex))
        traceback.print_exc()
        print("======================")
        return None