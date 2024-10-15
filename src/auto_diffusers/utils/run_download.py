import os
import requests
import time
import queue
import threading
from tqdm.auto import tqdm


class download:
    """
    Example:
    if __name__ == "__main__":
        file_path = download(url=<url>,save_path=<save_path>)

    Args:
    desc:str = "Running",
    fin_desc = "",
    pofix:str="",
    fin_pofix= "Finish!",
    desc_dot:bool = False,
    pofix_dot:bool = False

    """

    def __init__(
            self,
            url: str,
            save_path: str,
            desc: str = "",
            fin_desc = "",
            pofix: str = "",
            fin_pofix = "Finish!",
            desc_dot: bool = False,
            pofix_dot: bool = False,
            **ex_word
            ):
        super().__init__()
        self.url = url
        self.save_path = save_path
        self.desc = desc
        self.pofix = pofix
        self.fin_desc = fin_desc
        self.fin_pofix = fin_pofix
        self.desc_dot = desc_dot
        self.pofix_dot = pofix_dot
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
        self.response = self.response_get(url)
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        self.tqdm_obj = tqdm.wrapattr(
            open(save_path, "wb"),
            "write",
            miniters=1,
            desc=desc,
            postfix=pofix,
            total=int(self.response.headers.get('content-length', 0)),
            unit_scale=True,
            **ex_word
            )
      
        self.dot_thread.start()
        with self.tqdm_lock:
            for chunk in self.response.iter_content(chunk_size=4096):
                self.tqdm_obj.write(chunk)

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

    
    def response_get(self,url):
        try:
            response = requests.get(url, stream=True)    
            response.raise_for_status()
        except requests.HTTPError:
            raise requests.HTTPError(f"Invalid URL: {response.status_code}")
        return response
    

    def prosess_dot(self):
        """
        NOTE:
        set_description_str and set_description_str are not used,
        because they cannot be used in the case of the download_with_bar function.
        """
        self.dot_count = 0
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
                    self.tqdm_obj.refresh() #type: ignore

                    if self.stop_dot.is_set():
                        break
                    time.sleep(0.5)