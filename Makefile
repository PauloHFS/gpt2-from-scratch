# Makefile for the gpt2-from-scratch project

# --- Variáveis ---
PYTHON := python3 # Define o interpretador Python a ser usado.
SRC_DIR := src # Define o diretório do código-fonte.

# --- Configuração do Rclone ---
# Nome do seu remote configurado no rclone (ex: gdrive)
RCLONE_REMOTE := gdrive-ufcg
# Caminho para os dados locais que serão sincronizados
LOCAL_DATA_PATH := ./gutenberg
# Caminho no remote para onde os dados serão enviados
REMOTE_DATA_PATH := datasets/gutenberg

.PHONY: help
help:
	@echo "Comandos disponíveis:"
	@echo "  install      Instala o projeto e as dependências de desenvolvimento."
	@echo "  format       Formata o código-fonte com black e ruff."
	@echo "  lint         Verifica o código com ruff."
	@echo "  test         Executa os testes unitários com pytest."
	@echo "  quality      Executa format, lint e test em sequência."
	@echo "  clean        Remove arquivos temporários do Python."
	@echo "  data         (Placeholder) Baixa o dataset do Gutenberg."
	@echo "  sync-data    Sincroniza a pasta de dados local para o remote via rclone."

.PHONY: install
install:
	@echo "--- Instalando o pacote em modo editável e dependências de desenvolvimento ---"
	$(PYTHON) -m pip install -e .[dev]

.PHONY: format
format:
	@echo "--- Formatando o código com black e ruff ---"
	$(PYTHON) -m black $(SRC_DIR)
	$(PYTHON) -m ruff format $(SRC_DIR)
	$(PYTHON) -m ruff check $(SRC_DIR) --fix

.PHONY: lint
lint:
	@echo "--- Verificando o código com ruff ---"
	$(PYTHON) -m ruff check $(SRC_DIR)

.PHONY: test
test:
	@echo "--- Executando testes com pytest ---"
	$(PYTHON) -m pytest

.PHONY: quality
quality: format lint test
	@echo "--- Verificação de qualidade completa ---"

.PHONY: clean
clean:
	@echo "--- Removendo arquivos temporários ---"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf .ruff_cache

.PHONY: data
data:
	@echo "--- Baixando o dataset (placeholder) ---"
	git clone https://github.com/pgcorpus/gutenberg.git && \
	cd gutenberg && \
	python3 -m venv venv && \
	. venv/bin/activate && \
	pip install -r requirements.txt && \
	python get_data.py && \
	cd ..

.PHONY: sync-data
sync-data:
	@echo "--- Sincronizando $(LOCAL_DATA_PATH) para $(RCLONE_REMOTE):$(REMOTE_DATA_PATH) ---"
	rclone sync $(LOCAL_DATA_PATH) $(RCLONE_REMOTE):$(REMOTE_DATA_PATH) --progress --drive-pacer-min-sleep 10ms