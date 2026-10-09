#!/usr/bin/env bash
# Source this file so the exported variables remain in the calling shell:
#   source ./load_lsgemu_llm_key.sh
#   source ./load_lsgemu_llm_key.sh --store
#   source ./load_lsgemu_llm_key.sh --key-file /secure/path/dashscope_api_key

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf '%s\n' "error: run 'source ./load_lsgemu_llm_key.sh', not this script directly" >&2
    exit 2
fi

_lsgemu_loader_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_lsgemu_loader_config="${_lsgemu_loader_root}/LLM.env.yaml"
_lsgemu_loader_key_file="${LSGEMU_DASHSCOPE_KEY_FILE:-${XDG_CONFIG_HOME:-${HOME}/.config}/lsgemu/dashscope_api_key}"
_lsgemu_loader_store=0
_lsgemu_loader_error=""

while (($#)); do
    case "$1" in
        --store)
            _lsgemu_loader_store=1
            ;;
        --key-file)
            shift
            if (($# == 0)); then
                _lsgemu_loader_error="--key-file requires a path"
                break
            fi
            _lsgemu_loader_key_file="$1"
            ;;
        --config)
            shift
            if (($# == 0)); then
                _lsgemu_loader_error="--config requires a path"
                break
            fi
            _lsgemu_loader_config="$1"
            ;;
        -h|--help)
            printf '%s\n' \
                "source ./load_lsgemu_llm_key.sh [--store] [--key-file PATH] [--config PATH]" \
                "" \
                "Without --store, an existing DASHSCOPE_API_KEY is reused, then PATH is read," \
                "then an interactive hidden prompt is used. --store writes the prompted key to" \
                "PATH with mode 600. The default PATH is:" \
                "  \${XDG_CONFIG_HOME:-\$HOME/.config}/lsgemu/dashscope_api_key"
            unset _lsgemu_loader_root _lsgemu_loader_config _lsgemu_loader_key_file
            unset _lsgemu_loader_store _lsgemu_loader_error
            return 0
            ;;
        *)
            _lsgemu_loader_error="unknown option: $1"
            break
            ;;
    esac
    shift
done

if [[ -n "${_lsgemu_loader_error}" ]]; then
    printf 'error: %s\n' "${_lsgemu_loader_error}" >&2
    unset _lsgemu_loader_root _lsgemu_loader_config _lsgemu_loader_key_file
    unset _lsgemu_loader_store _lsgemu_loader_error
    return 2
fi

if [[ ! -r "${_lsgemu_loader_config}" ]]; then
    printf 'error: LLM configuration is not readable: %s\n' "${_lsgemu_loader_config}" >&2
    unset _lsgemu_loader_root _lsgemu_loader_config _lsgemu_loader_key_file
    unset _lsgemu_loader_store _lsgemu_loader_error
    return 2
fi

if ((_lsgemu_loader_store)); then
    if [[ ! -t 0 ]]; then
        printf '%s\n' "error: --store requires an interactive terminal" >&2
        unset _lsgemu_loader_root _lsgemu_loader_config _lsgemu_loader_key_file
        unset _lsgemu_loader_store _lsgemu_loader_error
        return 2
    fi
    read -r -s -p "DashScope API key: " DASHSCOPE_API_KEY
    printf '\n'
    if [[ -z "${DASHSCOPE_API_KEY}" ]]; then
        printf '%s\n' "error: empty API key" >&2
        unset DASHSCOPE_API_KEY
        unset _lsgemu_loader_root _lsgemu_loader_config _lsgemu_loader_key_file
        unset _lsgemu_loader_store _lsgemu_loader_error
        return 2
    fi
    _lsgemu_loader_key_dir="$(dirname -- "${_lsgemu_loader_key_file}")"
    if ! mkdir -p -- "${_lsgemu_loader_key_dir}"; then
        printf 'error: cannot create credential directory: %s\n' "${_lsgemu_loader_key_dir}" >&2
        unset DASHSCOPE_API_KEY _lsgemu_loader_key_dir
        unset _lsgemu_loader_root _lsgemu_loader_config _lsgemu_loader_key_file
        unset _lsgemu_loader_store _lsgemu_loader_error
        return 2
    fi
    chmod 700 -- "${_lsgemu_loader_key_dir}"
    _lsgemu_loader_old_umask="$(umask)"
    umask 077
    if ! printf '%s\n' "${DASHSCOPE_API_KEY}" > "${_lsgemu_loader_key_file}"; then
        umask "${_lsgemu_loader_old_umask}"
        printf 'error: cannot write credential file: %s\n' "${_lsgemu_loader_key_file}" >&2
        unset DASHSCOPE_API_KEY _lsgemu_loader_key_dir _lsgemu_loader_old_umask
        unset _lsgemu_loader_root _lsgemu_loader_config _lsgemu_loader_key_file
        unset _lsgemu_loader_store _lsgemu_loader_error
        return 2
    fi
    umask "${_lsgemu_loader_old_umask}"
    chmod 600 -- "${_lsgemu_loader_key_file}"
    _lsgemu_loader_source="stored_key_file"
elif [[ -n "${DASHSCOPE_API_KEY:-}" ]]; then
    _lsgemu_loader_source="existing_environment"
elif [[ -r "${_lsgemu_loader_key_file}" ]]; then
    mapfile -t _lsgemu_loader_key_lines < "${_lsgemu_loader_key_file}"
    if ((${#_lsgemu_loader_key_lines[@]} != 1)) || [[ -z "${_lsgemu_loader_key_lines[0]}" ]]; then
        printf 'error: credential file must contain exactly one non-empty line: %s\n' "${_lsgemu_loader_key_file}" >&2
        unset _lsgemu_loader_key_lines
        unset _lsgemu_loader_root _lsgemu_loader_config _lsgemu_loader_key_file
        unset _lsgemu_loader_store _lsgemu_loader_error
        return 2
    fi
    DASHSCOPE_API_KEY="${_lsgemu_loader_key_lines[0]}"
    _lsgemu_loader_source="key_file"
    unset _lsgemu_loader_key_lines
elif [[ -t 0 ]]; then
    read -r -s -p "DashScope API key (session only): " DASHSCOPE_API_KEY
    printf '\n'
    if [[ -z "${DASHSCOPE_API_KEY}" ]]; then
        printf '%s\n' "error: empty API key" >&2
        unset DASHSCOPE_API_KEY
        unset _lsgemu_loader_root _lsgemu_loader_config _lsgemu_loader_key_file
        unset _lsgemu_loader_store _lsgemu_loader_error
        return 2
    fi
    _lsgemu_loader_source="interactive_session"
else
    printf 'error: DASHSCOPE_API_KEY is unset and credential file is unavailable: %s\n' "${_lsgemu_loader_key_file}" >&2
    unset _lsgemu_loader_root _lsgemu_loader_config _lsgemu_loader_key_file
    unset _lsgemu_loader_store _lsgemu_loader_error
    return 2
fi

export DASHSCOPE_API_KEY
export LSGEMU_LLM_CONFIG="${_lsgemu_loader_config}"

printf 'LSGEmu LLM: credential=present source=%s\n' "${_lsgemu_loader_source}"
printf 'LSGEmu LLM config: %s\n' "${LSGEMU_LLM_CONFIG}"

unset _lsgemu_loader_key_dir _lsgemu_loader_old_umask _lsgemu_loader_source
unset _lsgemu_loader_root _lsgemu_loader_config _lsgemu_loader_key_file
unset _lsgemu_loader_store _lsgemu_loader_error
