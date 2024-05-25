import os

import numpy as np
import telegram
from keras.src.preprocessing import image
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackQueryHandler, ConversationHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from create_model import model
from use_model import find_object
from routes import create_route
import urllib.parse

token = "6552098456:AAFKoTSz1t3NKFdCXRU3DWybP-K1ycX3t9g"
object_name = ''

# Define states
AWAITING_RESPONSE = 1

# Define a function to handle the /start command
def start(update, context):
    update.message.reply_text("Привет, пришли мне фото достопримечательности ;)")

# Define a function to handle the /stop command
def stop(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Был рад помочь, если буду нужен снова просто напиши /start!")

# Define a function to handle photo messages
def handle_photo(update, context):
    # Get the photo file ID
    photo_file_id = update.message.photo[-1].file_id
    photo_file = context.bot.get_file(photo_file_id)
    # Download photo
    photo_file.download('./test_dir/test.jpg')

    update.message.reply_text("Принято, дай мне 5 сек :)")

# Функция для детектирования достопримечательностей
def cv2_imshow(img):
    pass


def detect_landmarks(image_path, cv2=None):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Применяем детектор Хаара
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    landmarks = cascade.detectMultiScale(gray, 1.1, 4)

    # Находим достопримечательность с наибольшей площадью
    largest_landmark = None
    largest_area = 0
    for (x, y, w, h) in landmarks:
        area = w * h
        if area > largest_area:
            largest_area = area
            largest_landmark = (x, y, w, h)

    # Выводим изображение с выделенной достопримечательностью
    if largest_landmark is not None:
        (x, y, w, h) = largest_landmark
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2_imshow(img)  # Используем cv2_imshow
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return largest_landmark, img

# Функция для предсказания типа достопримечательности
def predict_landmark_type(landmark, img, cv2=None, dataset_path=None):
    if landmark is None:
        return "Не найдено достопримечательностей"

    (x, y, w, h) = landmark
    cropped_image = img[y:y+h, x:x+w]

    # Преобразуем изображение для модели
    resized_image = cv2.resize(cropped_image, (224, 224))
    image_array = image.img_to_array(resized_image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Получаем предсказание модели
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)

    # Возвращаем название достопримечательности
    class_labels = os.listdir(dataset_path)
    return class_labels[predicted_class]
    
    model_response = find_object('./test_dir/test.jpg')

    global object_name
    object_name = model_response.get('name')

    context.bot.send_message(chat_id=update.message.chat_id, text=f"Кажется, это {object_name} {model_response.get('link')}")
    context.bot.send_location(chat_id=update.message.chat_id, latitude=model_response.get('latitude'), longitude=model_response.get('longitude'))
    # Add buttons after sending location
    keyboard = [[InlineKeyboardButton("Построить маршрут", callback_data='marshrut'), InlineKeyboardButton("Stop", callback_data='stop')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text("Хочешь построить маршрут к другим достопримечательностям нашего города?", reply_markup=reply_markup)

# Define a function to handle button clicks
def button_click(update, context):
    query = update.callback_query
    if query.data == 'stop':
        stop(update, context)
    if query.data == 'marshrut':
        keyboard = [
            [
                InlineKeyboardButton("2", callback_data='2'),
                InlineKeyboardButton("3", callback_data='3'),
                InlineKeyboardButton("4", callback_data='4'),
                InlineKeyboardButton("5", callback_data='5')
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        # Send message with the choices and markup
        context.bot.send_message(chat_id=query.message.chat_id, text="Сколько еще достопримечательностей хочешь посмотреть?", reply_markup=reply_markup)
    if query.data in ['2', '3', '4', '5']:
        global object_name
        visited, total_distance, hours, minutes = create_route(object_name, int(query.data) + 1)
        context.bot.send_message(chat_id=query.message.chat_id, text=f"{' -> '.join(visited)}.")
        context.bot.send_message(chat_id=query.message.chat_id, text=f"Общая дистанция: {total_distance:.2f} км.")
        context.bot.send_message(chat_id=query.message.chat_id, text=f"Общее время: {hours} часов и {minutes} минут.")
        stop(update, context)

def main():
    # Create an Updater object with your bot's token
    updater = Updater(token, use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # Register a command handler for the /start command
    dp.add_handler(CommandHandler("start", start))

    # Register a command handler for the /stop command
    dp.add_handler(CommandHandler("stop", stop))

    # Register a message handler for photo messages
    dp.add_handler(MessageHandler(Filters.photo, handle_photo))

    # Register a handler for button clicks
    dp.add_handler(CallbackQueryHandler(button_click))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C
    updater.idle()

if __name__ == '__main__':
    main()
