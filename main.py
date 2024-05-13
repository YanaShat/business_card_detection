from imutils.perspective import four_point_transform
from ultralytics import YOLO
from autocorrect import Speller
import telebot
import requests
import mysql.connector
import numpy as np
import pandas as pd
import pytesseract
import argparse
import imutils
import cv2
import re

BOT_TOKEN='6708684228:AAHMh5eMWTRoVJmSQfca_KsvN0cDevw56Y4'

bot = telebot.TeleBot(token='6708684228:AAHMh5eMWTRoVJmSQfca_KsvN0cDevw56Y4')

@bot.message_handler(content_types=['text', 'photo', 'file'])
def start(message):
    if message.content_type == 'photo' or message.content_type == 'file':
        bot.send_message(message.from_user.id, "I received message")
        photo = message.photo[-1]  # Get the last (highest resolution) photo size
        file_id = photo.file_id
        # bot.send_photo(message.from_user.id, file_id)

        file_info = bot.get_file(photo.file_id)
        file_path = file_info.file_path
        file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"

        response = requests.get(file_url)
        image_array = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        # Save the image locally
        cv2.imwrite("received_image.jpg", image)

        # You can further process the image here if needed

        # Send a message indicating successful processing
        bot.send_message(message.from_user.id, "Image processed successfully!")

        try:

            # # Подключение к удаленному MySQL серверу
            # mydb = mysql.connector.connect(
            #   host="100.114.35.34",
            #   user="pidor",
            #   password="Password12!@",
            #   database="BASE_FOR_TEST"  # Используйте вашу базу данных
            # )
            #
            # # Создание курсора и выполнение запросов
            # cursor = mydb.cursor()


            orig = cv2.imread("received_image.jpg")
            image = orig.copy()
            image = imutils.resize(image, height=800)
            ratio = orig.shape[1] / float(image.shape[1])

            model = YOLO('nanobest.pt')
            names = model.model.names
            results = model.predict(image, conf=0.4)  #Adjust conf threshold
            contours = results[0].masks.xy

            # initialize a contour that corresponds to the business card outline
            cardCnt = None
            # loop over the contours
            for c in contours:
              # approximate the contour
              peri = cv2.arcLength(c, True)
              approx = cv2.approxPolyDP(c, 0.02 * peri, True)
              # if this is the first contour we've encountered that has four
              # vertices, then we can assume we've found the business card
              if len(approx) == 4:
                cardCnt = approx
                break
            # if the business card contour is empty then our script could not
            # find the  outline of the card, so raise an error
            if cardCnt is None:
              raise Exception(("Could not find receipt outline. "
                "Try debugging your edge detection and contour steps."))
            # check to see if we should draw the contour of the business card
            # on the image and then display it to our screen

            cardCnt = np.array(cardCnt).reshape((-1,1,2)).astype(np.int32)

            # if args["debug"] > 0:
            #   output = image.copy()
            #   cv2.drawContours(output, [cardCnt], -1, (0, 255, 0), 2)
            #   cv2.imshow("Business Card Outline", output)
            #   cv2.waitKey(0)


            # apply a four-point perspective transform to the *original* image to
            # obtain a top-down bird's-eye view of the business card
            card = four_point_transform(orig, cardCnt.reshape(4, 2) * ratio)
            # show transformed image

            #cv2.imshow("Business Card Transform", card)
            #cv2.waitKey(0)

            # convert the business card from BGR to RGB channel ordering and then
            # OCR it
            rgb = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)

            pytesseract.pytesseract.tesseract_cmd = r'C:\Users\beedz\AppData\Local\Tesseract-OCR\tesseract.exe'
            EngText = pytesseract.image_to_string(rgb)
            RusText = pytesseract.image_to_string(rgb, lang='rus')

            # Создаем объекты Speller для английского и русского языков
            en_spell = Speller(lang='en')
            ru_spell = Speller(lang='ru')

            # Исправляем слова на английском языке
            corrected_eng_text = [en_spell(word) for word in EngText.split()]
            corrected_eng_text = ' '.join(corrected_eng_text)

            # Исправляем слова на русском языке
            corrected_rus_text = [ru_spell(word) for word in RusText.split()]
            corrected_rus_text = ' '.join(corrected_rus_text)

            # use regular expressions to parse out phone numbers and email
            # addresses from the business card
            phoneNums = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', corrected_eng_text)
            emails = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", corrected_eng_text)
            # attempt to use regular expressions to parse out names/titles (not
            # necessarily reliable)
            nameExp = r"^[\w'\-,.][^0-9_!¡?÷?¿/\\+=@#$%ˆ&*(){}|~<>;:[\]]{2,}"
            ru_names = re.findall(nameExp, corrected_rus_text)
            eng_names = re.findall(nameExp, corrected_eng_text)

            # show the phone numbers header
            print("PHONE NUMBERS")
            print("=============")
            # loop over the detected phone numbers and print them to our terminal
            for num in phoneNums:
              print(num.strip())
            # show the email addresses header
            print("\n")
            print("EMAILS")
            print("======")
            # loop over the detected email addresses and print them to our
            # terminal
            for email in emails:
              print(email.strip())
            # show the name/job title header
            print("\n")
            print("NAME/JOB TITLE")
            print("==============")
            # loop over the detected name/job titles and print them to our
            # terminal
            # for ru_name in ru_names:
            #   print(ru_name.strip())

            for eng_name in eng_names:
              print(eng_name.strip())

            # print(corrected_eng_text)
            df_information_from_cards = pd.DataFrame(
                {
                    "Name/Job title": [', '.join(eng_names)],
                    "Emails": [', '.join(emails)],
                    "Phone Number": [', '.join(phoneNums)],
                    "Corrected English Text": [corrected_eng_text],
                    "Corrected Russian Text": [corrected_rus_text]
                }
            )
            for inf in df_information_from_cards:
                print(df_information_from_cards[inf])
            bot.send_message(message.from_user.id, "Detection was sucessfully done!")
        except Exception as e:
            bot.send_message(message.from_user.id, "Something went wrong")


        # # Пример выполнения запроса INSERT
        # for index, row in df_information_from_cards.iterrows():
        #     name = row["Name/Job title"]
        #     email = row["Emails"]
        #     phone = row["Phone Number"]
        #     corrected_eng_text = row["Corrected English Text"]
        #     corrected_rus_text = row["Corrected Russian Text"]
        #
        #     # Выполнение запроса INSERT
        #     sql = "INSERT INTO table_for_test (name, email, phone, corrected_eng_text, corrected_rus_text) VALUES (%s, %s, %s, %s, %s)"
        #     val = (name, email, phone, corrected_eng_text, corrected_rus_text)
        #     cursor.execute(sql, val)
        #
        #
        # # Подтверждение выполнения изменений
        # mydb.commit()
        #
        # # Закрытие курсора и соединения
        # cursor.close()
        # mydb.close()

    else:
        bot.send_message(message.from_user.id, "I don't received message")

bot.polling(none_stop=True, interval=0)
