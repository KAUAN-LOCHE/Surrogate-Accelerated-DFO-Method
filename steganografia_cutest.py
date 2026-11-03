#!/usr/bin/env python3
"""Esteganografia LSB para esconder mensagem em histórico de convergência de funções matemáticas de CUTEST Collection.

Modos:
  encode       Esconde texto em uma imagem PNG existente.
  decode       Extrai texto de uma imagem PNG produzida por este script.
  plot-history Gera um gráfico de um histórico .npy e esconde o texto nele.

O método altera somente o bit menos significativo dos canais RGB da imagem.
Não altera arquivos .npy, resultados JSON, a função objetivo ou o otimizador.
"""

from __future__ import annotations

import argparse
import io
import struct
import sys
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


MAGIC = b"STG1"
# Cabeçalho: assinatura de 4 bytes, tamanho da mensagem (uint32) e CRC32 (uint32).
HEADER_FORMAT = ">4sII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


def _message_from_args(args: argparse.Namespace) -> str:
    """Obtém a mensagem de --message ou de --message-file, sempre em UTF-8."""
    if getattr(args, "message", None) is not None:
        return args.message
    message_file = getattr(args, "message_file", None)
    if message_file is not None:
        try:
            return Path(message_file).read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"Não foi possível ler o arquivo da mensagem: {exc}") from exc
    raise ValueError("Informe --message ou --message-file.")


def _image_to_rgb_array(image: Image.Image) -> np.ndarray:
    """Converte a imagem em RGB uint8, que é o formato usado para transportar bits."""
    return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def capacity_bytes(image: Image.Image) -> int:
    """Capacidade máxima aproximada em bytes de texto, descontando o cabeçalho."""
    rgb = _image_to_rgb_array(image)
    return max(0, (rgb.size // 8) - HEADER_SIZE)


def embed_text(image: Image.Image, message: str, output_path: str | Path) -> int:
    """Oculta uma mensagem UTF-8 nos bits menos significativos dos canais RGB.

    Retorna o número de bytes UTF-8 embutidos. A saída é salva como PNG sem perdas.
    """
    message_bytes = message.encode("utf-8")
    if len(message_bytes) > 0xFFFFFFFF:
        raise ValueError("A mensagem excede o limite de tamanho do cabeçalho (4 GiB).")

    rgb = _image_to_rgb_array(image)
    flat = rgb.reshape(-1)
    payload_capacity = max(0, (flat.size // 8) - HEADER_SIZE)
    if len(message_bytes) > payload_capacity:
        raise ValueError(
            f"Mensagem grande demais: {len(message_bytes)} bytes, mas a imagem comporta "
            f"no máximo {payload_capacity} bytes. Use uma imagem maior ou uma mensagem menor."
        )

    checksum = zlib.crc32(message_bytes) & 0xFFFFFFFF
    header = struct.pack(HEADER_FORMAT, MAGIC, len(message_bytes), checksum)
    payload = header + message_bytes
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")

    # Apenas os bits menos significativos são modificados; os demais bits de cada canal
    # permanecem idênticos. A ordem é determinística para permitir a extração.
    flat[:bits.size] = (flat[:bits.size] & 0xFE) | bits
    stego_image = Image.fromarray(flat.reshape(rgb.shape))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    stego_image.save(output, format="PNG")
    return len(message_bytes)


def extract_text(image: Image.Image) -> str:
    """Extrai mensagem e valida a assinatura, o tamanho e o CRC32."""
    rgb = _image_to_rgb_array(image)
    flat = rgb.reshape(-1)
    header_bits_count = HEADER_SIZE * 8
    if flat.size < header_bits_count:
        raise ValueError("Imagem pequena demais para conter o cabeçalho esteganográfico.")

    header_bits = flat[:header_bits_count] & 1
    header_bytes = np.packbits(header_bits, bitorder="big").tobytes()
    magic, message_length, expected_crc = struct.unpack(HEADER_FORMAT, header_bytes)
    if magic != MAGIC:
        raise ValueError(
            "Assinatura esteganográfica não encontrada. Verifique se a imagem foi criada "
            "por este script e se não foi redimensionada ou recomprimida."
        )

    total_bits = (HEADER_SIZE + message_length) * 8
    if total_bits > flat.size:
        raise ValueError("Cabeçalho inválido: tamanho declarado maior que a capacidade da imagem.")

    all_bits = flat[:total_bits] & 1
    payload_bytes = np.packbits(all_bits, bitorder="big").tobytes()
    message_bytes = payload_bytes[HEADER_SIZE:HEADER_SIZE + message_length]
    actual_crc = zlib.crc32(message_bytes) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise ValueError(
            "A verificação de integridade falhou. A imagem pode ter sido alterada, "
            "redimensionada ou recomprimida após a inserção."
        )

    try:
        return message_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("A carga recuperada não é um texto UTF-8 válido.") from exc


def _make_history_plot(history_path: str | Path) -> Image.Image:
    """Cria em memória um gráfico de convergência a partir de um histórico .npy."""
    path = Path(history_path)
    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Não foi possível carregar o histórico {path}: {exc}") from exc

    values = np.asarray(values)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("O histórico precisa ser um vetor .npy não vazio.")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("O histórico .npy deve conter valores numéricos.")

    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("O histórico não contém valores finitos para plotar.")

    evaluations = np.arange(1, values.size + 1)
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(evaluations[finite], values[finite], linewidth=1.5)
    ax.set_title(f"Histórico de convergência — {path.stem}")
    ax.set_xlabel("Avaliações da função")
    ax.set_ylabel("Melhor valor acumulado de f(x)")
    ax.grid(True, linestyle="--", alpha=0.35)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="PNG", dpi=160)
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as generated:
        return generated.convert("RGB").copy()


def create_stego_from_history(history_path: str | Path, message: str, output_path: str | Path) -> int:
    """API pública para o benchmark gerar um gráfico e embutir texto nele.

    É usada pelo gancho opcional de benchmark_pycutest_simplex_stego.py.
    Retorna o número de bytes UTF-8 embutidos.
    """
    image = _make_history_plot(history_path)
    embedded_bytes = embed_text(image, message, output_path)
    with Image.open(output_path) as saved:
        max_payload = capacity_bytes(saved)
    print(f"Imagem esteganográfica salva em: {output_path}")
    print(f"Mensagem embutida: {embedded_bytes} bytes UTF-8.")
    print(f"Capacidade aproximada de mensagem desta imagem: {max_payload} bytes.")
    print("Importante: mantenha a imagem em PNG; não redimensione nem recomprima antes de extrair.")
    return embedded_bytes


def _add_message_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--message", help="Texto a ocultar (UTF-8).")
    group.add_argument("--message-file", help="Arquivo de texto UTF-8 com a mensagem.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Demonstração de esteganografia LSB em imagens de resultados CUTEst."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    encode_parser = subparsers.add_parser("encode", help="Oculta texto em uma imagem existente.")
    encode_parser.add_argument("--image", required=True, help="Imagem de entrada.")
    _add_message_arguments(encode_parser)
    encode_parser.add_argument("--output", required=True, help="Caminho da imagem PNG de saída.")

    decode_parser = subparsers.add_parser("decode", help="Extrai texto de uma imagem esteganográfica.")
    decode_parser.add_argument("--image", required=True, help="Imagem PNG que contém a mensagem.")

    plot_parser = subparsers.add_parser(
        "plot-history", help="Gera um gráfico de histórico .npy e oculta a mensagem nele."
    )
    plot_parser.add_argument("--history", required=True, help="Arquivo .npy salvo pelo benchmark.")
    _add_message_arguments(plot_parser)
    plot_parser.add_argument("--output", required=True, help="Caminho da imagem PNG de saída.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "decode":
            with Image.open(args.image) as image:
                message = extract_text(image)
            print("Mensagem extraída (UTF-8):")
            print(message)
            return 0

        message = _message_from_args(args)
        if args.command == "encode":
            with Image.open(args.image) as source:
                image = source.convert("RGB")
            embedded_bytes = embed_text(image, message, args.output)
        elif args.command == "plot-history":
            create_stego_from_history(args.history, message, args.output)
            return 0
        else:  # Proteção defensiva; argparse restringe os comandos válidos.
            parser.error("Comando desconhecido.")
            return 2

        with Image.open(args.output) as saved:
            max_payload = capacity_bytes(saved)
        print(f"Imagem esteganográfica salva em: {args.output}")
        print(f"Mensagem embutida: {embedded_bytes} bytes UTF-8.")
        print(f"Capacidade aproximada de mensagem desta imagem: {max_payload} bytes.")
        print("Importante: mantenha a imagem em PNG; não redimensione nem recomprima antes de extrair.")
        return 0

    except (OSError, ValueError, struct.error) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
