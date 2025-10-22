<?php

namespace App\Support;

use JsonException;

class SafeJson
{
    private const DEFAULT_FLAGS = JSON_HEX_TAG
        | JSON_HEX_APOS
        | JSON_HEX_AMP
        | JSON_HEX_QUOT;

    /**
     * Encode the given value into JSON safely, substituting invalid UTF-8 sequences.
     *
     * This mirrors Laravel's built-in @json behaviour while forcing invalid byte
     * sequences to be replaced so rendering never fails on malformed input.
     *
     * @throws JsonException
     */
    public static function encode(mixed $value, int $flags = 0, int $depth = 512): string
    {
        $options = self::DEFAULT_FLAGS | JSON_INVALID_UTF8_SUBSTITUTE | $flags;

        $json = json_encode($value, $options, $depth);

        if ($json === false) {
            throw new JsonException(json_last_error_msg(), json_last_error());
        }

        return $json;
    }
}