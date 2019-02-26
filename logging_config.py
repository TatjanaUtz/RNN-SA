"""Configurations for logging."""
import logging

LOG_FILE_NAME = "results.log"


def init_logging():
    """Initializes logging.

    Configures logging. Error messages are logged to the 'error.log' file. All messages are logged
    to the console.
    """
    # create logger for 'RNN-SA' project
    logger = logging.getLogger('RNN-SA')
    logger.setLevel(logging.DEBUG)

    # create file handler which logs error messages
    log_file_handler = logging.FileHandler('error.log', mode='w+')
    log_file_handler.setLevel(logging.ERROR)

    # create console handler with a lower log level (e.g. debug or info)
    log_console_handler = logging.StreamHandler()
    log_console_handler.setLevel(logging.DEBUG)

    # create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file_handler.setFormatter(formatter)
    log_console_handler.setFormatter(formatter)

    # add the handler to the logger
    logger.addHandler(log_file_handler)
    logger.addHandler(log_console_handler)

    # clear log file for results
    log_file = open(LOG_FILE_NAME, 'w+')
    log_file.close()

    return logger


def log_results(name, hyperparams, res):
    """Log results of a recurrent neural network.

    Overview of the results of a recurrent neural network is printed.

    Args:
        name -- name of the recurrent network
        hyperparams -- dictionary of the used hyperparameters
        res -- dictionary with the results (e.g. F1 Score, Accuracy Score, Recall, Precision)
    """
    # create logger
    logger = logging.getLogger('RNN-SA.logging_config.py.log_results')

    # check input arguments
    if not isinstance(name, str):  # invalid argument for name
        logger.error("Invalid argument for 'name': is %s but must be <class 'str'>!", type(name))
        return
    if not isinstance(hyperparams, dict):  # invalid argument for hyperparameters
        logger.error("Invalid argument for 'hyperparameters': is %s but must be <class 'dict'>!",
                     type(hyperparams))
        return
    if not isinstance(res, dict):  # invalid argument for results
        logger.error("Invalid argument for 'results': %s but must be <class 'dict'>!",
                     type(res))
        return

    # log results to a file
    log_file = open(LOG_FILE_NAME, 'a+')  # open or create log file ('+') and append lines ('a')
    log_file.write("\n")
    title_string = "-------------------- " + name + " --------------------"
    log_file.write(title_string + "\n")

    # log hyperparameter
    log_file.write("------- Hyperparameter ------- \n")
    for key in hyperparams:  # loop over all hyperparameters
        log_file.write(key + ": " + str(hyperparams[key]) + "\n")
    log_file.write("\n")

    # log results
    log_file.write("----------- Results ----------- \n")
    for key in res:  # loop over all results
        log_file.write(key + ": " + str(res[key]) + "\n")
    log_file.write("\n")

    log_file.write("-" * len(title_string) + "\n")
    log_file.close()

    # Print results to console
    logger.info(title_string)
    for key in res:  # loop over all results
        logger.info("%s: %f", key, res[key])
    logger.info("%s \n", "-" * len(title_string))


if __name__ == "__main__":
    print("Main function of logging_config.py")
