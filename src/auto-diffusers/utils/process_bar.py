import os
import time
import queue
import threading
from tqdm.auto import tqdm

class ProcessBarRun:
    """
    Example:
    if __name__ == "__main__":
    bar = ProcessBarRun(total=11)

    for s in range(4):
        bar.bar_update(3)
        time.sleep(1)

    bar.bar_update(exit=True)

    Args:
    desc:str = "Running",
    default_desc:str = "Running",
    fin_desc = "",
    pofix:str="",
    default_pofix:str = "",
    fin_pofix= "Finish!",
    desc_dot:bool = False,
    pofix_dot:bool = False

    If the download_bar is true, the URL and save_path are required
    """

    def __init__(self,
                 total: int = 0,
                 desc: str = "",
                 default_desc: str = "",
                 fin_desc="",
                 pofix: str = "",
                 default_pofix: str = "",
                 fin_pofix="Finish!",
                 desc_dot: bool = False,
                 pofix_dot: bool = False,
                 download_bar:bool = False,
                 url:str = "",
                 save_path:str = "",
                 **ex_word):

        self.total = total
        self.default_desc = default_desc
        self.default_pofix = default_pofix
        self.fin_desc = fin_desc
        self.fin_pofix = fin_pofix
        self.desc_dot = desc_dot
        self.pofix_dot = pofix_dot
        self.ex_word = ex_word
        self.queue_obj = queue.Queue()
        self._count = 0
        self._run_count = 0
        self.dot_count = 0
        self.max_dots = 5
        self.desc = desc
        self.pofix = pofix
        self.base_desc_txt = ""
        self.base_pofix_txt = ""
        self.stop_event = threading.Event()
        self.stop_dot = threading.Event()
        self.tqdm_lock = threading.Lock()
        self.tqdm_obj = tqdm(total=total, desc=desc, postfix=pofix)
        self.arg_update(desc=desc, postfix=pofix, **ex_word)
        self.dot_thread = threading.Thread(target=self.prosess_dot)
        self.dot_thread.start()


    def __del__(self):
        self.stop()


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


    def stop(self):
        self.stop_event.set()
        self.stop_dot.set()
        self.tqdm_obj.n = self.total
        if self.fin_desc:
            setattr(self.tqdm_obj,"desc",self.fin_desc)
        if self.fin_pofix:
            setattr(self.tqdm_obj,"postfix",self.fin_pofix)
        self.tqdm_obj.refresh()
        self.tqdm_obj.close()


    def arg_update(self, *ex_word, **input_dict):
        for extra_word in ex_word:
            if isinstance(extra_word, dict):
                for _key, _value in extra_word.items():
                    setattr(self, _key, _value)
        for key, value in input_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


    def run_downlaod_with_bar(self,response):
        with self.tqdm_lock:
            for chunk in response.iter_content(chunk_size=4096):
                self.tqdm_obj.write(chunk)


    def prosess_dot(self,):
        """
        NOTE:
        set_description_str and set_description_str are not used,
        because they cannot be used in the case of the download_with_bar function.
        """
        self.dot_count = 0
        chenge_check = False
        while not self.stop_dot.is_set():
            for num in range(self.max_dots):
                with self.tqdm_lock:
                    dot_txt = "." * num
                    if self.desc and self.desc_dot:
                        desc_dot_txt = dot_txt
                    else:
                        desc_dot_txt = ""

                    if self.pofix and self.pofix_dot:
                        pofix_dot_txt = dot_txt
                    else:
                        pofix_dot_txt = ""

                    setattr(self.tqdm_obj, "desc", self.desc + desc_dot_txt)
                    setattr(self.tqdm_obj, "postfix", self.pofix + pofix_dot_txt)
                    self.tqdm_obj.refresh()

                    if self.stop_dot.is_set():
                        break
                    time.sleep(0.5)


    def bar_update(
            self,
            update_rate: int = 1,
            exit: bool = False,
            desc = None,
            pofix = None,
            desc_dot = None,
            pofix_dot= None):
        """
        args:
        desc : str
        pofix : str
        desc_dot : bool
        pofix_dot : bool
        """
        self.desc = desc or self.desc or self.default_desc
        self.pofix = pofix or self.pofix or self.default_pofix
        self.desc_dot = desc_dot or self.desc_dot
        self.pofix_dot = pofix_dot or self.pofix_dot
        self.tqdm_obj.update(int(update_rate))
        self.tqdm_obj.set_description_str(self.desc)
        self.tqdm_obj.set_postfix_str(self.pofix)
        self.tqdm_obj.refresh()
        if exit:
            self.stop()
