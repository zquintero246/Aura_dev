# ===== Laravel backend =====

# Imagen base con PHP y extensiones necesarias
FROM php:8.2-fpm

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    git \
    unzip \
    libpq-dev \
    libzip-dev \
    zip \
    libicu-dev \
    g++ \
    autoconf \
    pkg-config \
    libssl-dev \
    && docker-php-ext-install intl pdo pdo_pgsql \
    && pecl install mongodb \
    && docker-php-ext-enable mongodb

# Instalar Composer
COPY --from=composer:2 /usr/bin/composer /usr/bin/composer

# Crear directorio de trabajo
WORKDIR /var/www/html

# Copiar los archivos de Laravel
COPY . .

# Instalar dependencias PHP
RUN composer install --no-interaction --prefer-dist --optimize-autoloader

# Dar permisos a storage y bootstrap
RUN chown -R www-data:www-data storage bootstrap/cache
RUN chmod -R 775 storage bootstrap/cache

# Exponer el puerto 9000 para php-fpm
EXPOSE 9000

# Comando por defecto
CMD ["php-fpm"]
