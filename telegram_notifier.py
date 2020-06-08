# Uses this https://github.com/python-telegram-bot/python-telegram-bot

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters


def initialize(token):
    updater = Updater(token, use_context=True)
    # Get the dispatcher to register handlers
    dp = updater.dispatcher
    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    # log all errors
    dp.add_error_handler(error)
    # Start the Bot
    updater.start_polling()
    return updater


def start(update, context):
    """Send a message when the command /start is issued."""
    global tg_chat_id, tg_context
    tg_chat_id = update.message.chat_id
    tg_context = context
    update.message.reply_text('Hi! {}'.format(tg_chat_id))

def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)

tg_chat_id = None

def notify(msg):
    global tg_chat_id, tg_context
    print('Chat id {}'.format(tg_chat_id))
    if not (tg_chat_id and tg_context):
        return
    print(msg)
    tg_context.bot.send_message(tg_chat_id, msg)


