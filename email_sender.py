import smtplib
import ssl
from email.message import EmailMessage
from email.mime.image import MIMEImage


def send_email(filename: str, receiver_email: str = "an.s.gavrilenko@gmail.com",
               subject: str = "An email with attachment from Python4",
               body: str = "This is an email with attachment sent from Python",
               sender_email: str = "gmandrtest@gmail.com", password: str = "wsjpkirfwqvgbucx"):
    smtp_server = "smtp.gmail.com"

    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    # me == the sender's email address
    # family = the list of all recipients' email addresses
    msg['From'] = sender_email
    msg['To'] = receiver_email

    with open(filename, 'rb') as fp:
        img_data = fp.read()
        msg.add_attachment(img_data, maintype='image',
                           subtype='jpg', filename="file")

    text = msg.as_string()

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        try:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, text)
        except Exception as e:
            print(e)
        finally:
            server.quit()
