<?php

namespace App\Http\Controllers\Auth;

use App\Http\Controllers\Controller;
use App\Models\User;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Laravel\Socialite\Facades\Socialite;
use Laravel\Socialite\Two\User as SocialiteUser;
use Illuminate\Support\Str;

class SocialiteController extends Controller
{
    /**
     * List of all official Socialite providers supported.
     * This list is used for validation and database field naming.
     */
    protected $officialProviders = [
        'github',
        'google',
        'facebook',
        'gitlab',
        'bitbucket',
        'slack',
        'twitch',
        'twitter-openid', // Twitter (X) now uses the OpenID flow
        'linkedin-openid', // LinkedIn also uses the OpenID flow
    ];

    /**
     * Redirect the user to the OAuth Provider.
     *
     * @param string $provider
     * @return \Illuminate\Http\Response
     */
    public function redirectToProvider(string $provider)
    {
        if (!in_array($provider, $this->officialProviders) || !config("services.{$provider}")) {
             // Abort if the provider is not supported or misconfigured
             abort(404, "Socialite provider {$provider} not supported or configured.");
        }

        // Redirect to the social provider's authorization page
        return Socialite::driver($provider)->redirect();
    }

    /**
     * Obtain the user information from the OAuth Provider and handle login/creation.
     *
     * @param string $provider
     * @return \Illuminate\Http\Response
     */
    public function handleProviderCallback(string $provider)
    {
        if (!in_array($provider, $this->officialProviders)) {
             abort(404);
        }

        try {
            // Retrieve the user from the social provider
            $socialiteUser = Socialite::driver($provider)->user();
        } catch (\Exception $e) {
            // Handle exceptions (e.g., user denied access, invalid state/CSRF token)
            return redirect()->route('login')->withErrors(['email' => 'Authentication failed. Please try again or choose a different method.']);
        }

        // Centralized logic to find or create the user and assign the role
        $user = $this->findOrCreateUser($socialiteUser, $provider);

        // Log the user into the application
        Auth::login($user, true);

        return redirect()->intended('/admin/dashboard'); // Add your dashboard link
    }

    /**
     * Finds user by provider ID or email, creates if necessary, and assigns 'subscriber' role using Spatie.
     *
     * @param SocialiteUser $socialiteUser
     * @param string $provider
     * @return User
     */
    protected function findOrCreateUser(SocialiteUser $socialiteUser, string $provider): User
    {
        // Normalize the provider key for database column names (e.g., 'twitter-openid' becomes 'twitter_openid_id')
        $providerKey = str_replace('-', '_', $provider);
        $providerIdField = "{$providerKey}_id";

        // 1. Check if user already exists via the social provider's unique ID
        $user = User::where($providerIdField, $socialiteUser->getId())->first();

        if ($user) {
            // User exists and is linked to this social account.
            // DO NOT assign any role as per requirement.
            return $user;
        }

        // 2. Check if user exists via email (for account linking)
        $user = User::where('email', $socialiteUser->getEmail())->first();

        if ($user) {
            // User exists by email, link the social ID to the existing account.
            // DO NOT assign any role as per requirement.
            $user->update([
                $providerIdField => $socialiteUser->getId(),
            ]);
            return $user;
        }

        // 3. User does not exist, so create a new one.
        $user = User::create([
            'name' => $socialiteUser->getName() ?? $socialiteUser->getNickname() ?? 'New Social User',
            'email' => $socialiteUser->getEmail(),
            // Create a random password since social login is primary
            'password' => bcrypt(Str::random(24)),
            $providerIdField => $socialiteUser->getId(),
            // You may need to verify the email address here based on provider data
            'email_verified_at' => now(),
        ]);

        // SPATIE ROLE LOGIC: Assign 'subscriber' role only on first creation
        // IMPORTANT: Ensure the 'subscriber' role is seeded in your database!
        // $user->assignRole('subscriber');  // this is optional if using spatie role permission package

        return $user;
    }
}
