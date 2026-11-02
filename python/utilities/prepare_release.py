# Prepare the release library
import os
import sys
import shutil
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).parent.resolve()
PYTHON_DIR = ROOT.parent


SUPPORTED_GAMES = ('aoe2', 'aoe4', 'aom', 'sc2', 'wc3')


def compile_clean(name_game: str, disable_console: bool = True, finalize_folder: bool = False):
    """Compile an overlay program and clean the building files.

    Parameters
    ----------
    name_game          Name of the game.
    disable_console    True to disable the console output.
    finalize_folder    True to finalize the folder (copy additional files, zip, clean).
    """
    if name_game not in SUPPORTED_GAMES:
        raise ValueError(
            f'Unsupported game {name_game!r}. Expected one of {SUPPORTED_GAMES}.'
        )

    name_main_file = 'main_' + name_game  # name of the main python file (without path & extension)
    name_out_lib = name_game + '_overlay'  # name of the output library

    # resolve all build artifact/output paths against ROOT (not the caller's cwd)
    name_out_lib_path = ROOT / name_out_lib
    overlay_folder = name_out_lib_path / 'overlay'
    utilities_folder = overlay_folder / 'utilities'

    if name_out_lib_path.is_dir():
        raise FileExistsError(f'Output folder already exists: {name_out_lib_path}')
    name_out_lib_path.mkdir()
    overlay_folder.mkdir()
    utilities_folder.mkdir()

    icon = ROOT / '../../docs/assets/common/icon/salamander_sword_shield.ico'  # icon for the library
    icon = icon.resolve()

    include_common = (PYTHON_DIR / 'common').resolve()
    include_game = (PYTHON_DIR / name_game).resolve()
    include_docs_common = (ROOT / '../../docs/assets/common').resolve()
    include_docs_game = (ROOT / f'../../docs/assets/{name_game}').resolve()
    main_file = (PYTHON_DIR / f'{name_main_file}.py').resolve()

    # nuitka command to run
    command = [
        sys.executable,
        '-m', 'nuitka',
        '--standalone',
        '--plugin-enable=pyqt5',
        '--assume-yes-for-downloads',
        f'--windows-icon-from-ico={icon}',
        f'--include-data-file={include_common}/*.py=common/',
        f'--include-data-file={include_game}/*.py={name_game}/',
        f'--include-data-dir={include_docs_common}=docs/assets/common',
        f'--include-data-dir={include_docs_game}=docs/assets/{name_game}',
        '--include-package=omegaconf',
    ]

    if disable_console:  # disable the console
        command.append('--windows-disable-console')

    command.append(str(main_file))  # main file to compile

    subprocess.run(command, check=True, cwd=str(PYTHON_DIR))  # compilation

    dist_dir = PYTHON_DIR / f'{name_main_file}.dist'
    build_dir = PYTHON_DIR / f'{name_main_file}.build'

    # rename executable name
    (dist_dir / f'{name_main_file}.exe').rename(dist_dir / f'{name_out_lib}.exe')

    if not disable_console:  # rename for version with console
        (dist_dir / f'{name_out_lib}.exe').rename(dist_dir / f'{name_out_lib}_with_console.exe')

    # copy files in output directory
    shutil.copytree(dist_dir, overlay_folder, dirs_exist_ok=True)

    # move pictures in new directory
    out_docs = name_out_lib_path / 'docs'
    out_docs.mkdir(parents=True, exist_ok=True)
    shutil.move(str(overlay_folder / 'docs' / 'assets'), str(out_docs))

    # clean building files
    shutil.rmtree(build_dir)
    shutil.rmtree(dist_dir)

    # finalize the output folder
    if finalize_folder:
        # copy readme, changelog, license, version and requirements
        shutil.copy(ROOT / '../../Readme.md', name_out_lib_path)
        shutil.copy(ROOT / '../../Changelog.md', name_out_lib_path)
        shutil.copy(ROOT / '../../LICENSE', name_out_lib_path)
        shutil.copy(ROOT / '../../version.json', name_out_lib_path)
        shutil.copy(ROOT / 'requirements.txt', utilities_folder)

        # copy remaining source files
        shutil.copy(main_file, overlay_folder)
        shutil.copy(ROOT / 'prepare_release.py', utilities_folder)

        if name_game == 'aoe4':
            demo_data = PYTHON_DIR / 'demo_data'
            # Ship the same dual-band timer ROI used by recording acceptance.
            # The legacy 22px crop cannot see both floating timer bands.
            shutil.copy(
                demo_data / 'smart_hud_config_1280x720.json',
                overlay_folder / 'smart_hud_config.json',
            )
            shutil.copytree(
                demo_data / 'hud_templates_1280x720',
                overlay_folder / 'hud_templates_1280x720',
            )
            build_orders_out = overlay_folder / 'local_config' / 'aoe4' / 'build_orders'
            build_orders_out.mkdir(parents=True, exist_ok=True)
            for build_order_file in (demo_data / 'build_orders').glob('*.json'):
                shutil.copy(build_order_file, build_orders_out)

        # zip output folder
        shutil.make_archive(str(name_out_lib_path), 'zip', str(name_out_lib_path))
        shutil.rmtree(name_out_lib_path)  # clean non-zipped files


if __name__ == '__main__':
    # python prepare_release.py [name_game] (aoe2, aoe4, aom, sc2, wc3)
    if len(sys.argv) > 2:
        print(f'Usage: {sys.argv[0]} [{"|".join(SUPPORTED_GAMES)}]', file=sys.stderr)
        sys.exit(2)
    if len(sys.argv) != 2:  # compile all projects
        selected_games = list(SUPPORTED_GAMES)
    else:
        game_arg = sys.argv[1]
        if game_arg not in SUPPORTED_GAMES:
            print(
                f'Unsupported game {game_arg!r}. Expected one of {SUPPORTED_GAMES}.',
                file=sys.stderr,
            )
            sys.exit(2)
        selected_games = [game_arg]

    try:
        for selected_game in selected_games:
            compile_clean(name_game=selected_game, disable_console=True, finalize_folder=True)
    except Exception as exc:
        print(f'Release preparation failed: {exc}', file=sys.stderr)
        sys.exit(1)
