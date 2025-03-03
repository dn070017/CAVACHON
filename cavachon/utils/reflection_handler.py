import glob
import os
import re
from importlib import import_module
from pathlib import Path

from cavachon.environment.settings import Settings


class ReflectionHandler:
    @staticmethod
    def _import_class(path: Path, class_name: str):
        module_path = ".".join(
            path.parts[
                len(list(path.parts)) - list(path.parts)[::-1].index("cavachon") - 1 :
            ]
        )[:-3]
        mod = import_module(module_path)
        cls = getattr(mod, class_name)
        return cls

    @staticmethod
    def _camel_to_snake(name):
        snake = re.sub(
            r"([a-z])([A-Z])", r"\1_\2", name
        )  # Add underscore between lowercase and uppercase
        return snake.lower()

    @staticmethod
    def get_class_by_name(class_name: str, subdirectory: str = "", postfix: str = ""):
        class_snake_name = ReflectionHandler._camel_to_snake(class_name)
        postfix_snake_name = ReflectionHandler._camel_to_snake(postfix)
        if len(postfix_snake_name) != 0:
            postfix_snake_name = f"_{postfix_snake_name}"
        filename = f"{class_snake_name}{postfix_snake_name}.py"

        filenames = ReflectionHandler._get_filenames(filename, subdirectory)

        assert len(filenames) > 0, (
            f"ReflectionHandler could not find file named {filename}. Please check ",
            "the spelling and try again.",
        )
        assert len(filenames) == 1, (
            f"ReflectionHandler find more than one files named {filename}. Please check ",
            "the spelling and try again.",
        )
        return ReflectionHandler._import_class(filenames[0], f"{class_name}{postfix}")

    @staticmethod
    def _get_filenames(
        filename: str,
        subdirectory: str = "",
        max_depth=2,
        partial=False,
    ):
        pattern = f"*{filename}" if partial else f"{filename}"
        result = []
        for depth in range(max_depth):
            filename = (
                f"{Settings.src_path}/{subdirectory}" + "**/" * (depth + 1) + pattern
            )
            if len(glob.glob(filename)) != 0:
                result.append(
                    Path(os.path.join(Settings.src_path, glob.glob(filename)[0]))
                )
                break
        return result
