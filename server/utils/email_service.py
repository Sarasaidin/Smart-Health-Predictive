from azure.communication.email import EmailClient
import os
from dotenv import load_dotenv
# Load environment variables.
load_dotenv()


def send_email(recipient: str, subject: str, content: str, content_type: str = "plainText"):
    """
    Sends an email using Azure Communication Service.

    :param recipient: The email address of the recipient.
    :param subject: The subject of the email.
    :param content: The body of the email (either plain text or HTML).
    :param content_type: The type of content, 'plainText' or 'html'. Defaults to 'plainText'.
    :return: The result of the send operation.
    """
    try:
        # It's recommended to use environment variables for connection strings.
        connection_string = os.environ.get("AZURE_EMAIL_CONNECTION_STRING", "")
        sender_address = os.environ.get("AZURE_EMAIL_SENDER_ADDRESS")
        client = EmailClient.from_connection_string(connection_string)

        message_content = {
            "subject": subject,
        }
        if content_type == "html":
            message_content["html"] = content
        else:
            message_content["plainText"] = content

        message = {
            "senderAddress": sender_address,
            "recipients": {
                "to": [{"address": recipient}]
            },
            "content": message_content,
        }

        print(f"Sending email to: {recipient}")

        poller = client.begin_send(message)
        result = poller.result()

        print("Azure Email Result:")
        print(result)
        return result

    except Exception as ex:
        import traceback

        print("===== EMAIL ERROR =====")
        traceback.print_exc()
        print(ex)
        print("=======================")

    return None


if __name__ == '__main__':
    # Example of sending a plain text email
    send_email(
        recipient="test@gmail.com",
        subject="Test Email (Plain Text)",
        content="Hello world via email from the new function."
    )

    # Example of sending an HTML email
    html_content = """
    <html>
        <body>
            <h1>Hello world via email.</h1>
            <p>This is an HTML email from the new function.</p>
        </body>
    </html>
    """
    send_email(
        recipient="test@gmail.com",
        subject="Test Email (HTML)",
        content=html_content,
        content_type="html"
    )
