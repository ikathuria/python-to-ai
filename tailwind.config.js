// Tailwind build config for the lesson pages. Theme maps Tailwind's colour names onto the
// brand tokens in app/styles/brand.css, so lesson markup picks up the brand and dark mode.
// Rebuild after changing classes in app/pages/*.html:  npm run build:css
function family(name) {
	return {
		50: `var(--${name}-bg)`, 100: `var(--${name}-bg)`, 200: `var(--${name}-line)`, 300: `var(--${name}-line)`,
		400: `var(--${name}-solid)`, 500: `var(--${name}-solid)`, 600: `var(--${name}-solid)`,
		700: `var(--${name}-text)`, 800: `var(--${name}-text)`, 900: `var(--${name}-text)`,
	};
}

const theme = {
	theme: {
		extend: {
			colors: {
				white: 'var(--surface)',
				gray: {
					50: 'var(--bg)', 100: 'var(--case)', 200: 'var(--line)', 300: 'var(--line)',
					400: 'var(--muted)', 500: 'var(--muted)', 600: 'var(--muted)', 700: 'var(--muted)',
					800: 'var(--text)', 900: 'var(--screen)',
				},
				green: family('ok'), emerald: family('ok'), teal: family('ok'), lime: family('ok'),
				red: family('bad'), rose: family('bad'), pink: family('bad'),
				yellow: family('warn'), amber: family('warn'), orange: family('warn'),
				blue: family('info'), sky: family('info'), cyan: family('info'),
				purple: family('violet'), violet: family('violet'), fuchsia: family('violet'),
				indigo: {
					50: 'var(--tint)', 100: 'var(--tint)', 200: 'var(--tint-strong)', 300: 'var(--tint-strong)',
					400: 'var(--link)', 500: 'var(--link)', 600: 'var(--link)',
					700: 'var(--link-strong)', 800: 'var(--link-strong)', 900: 'var(--link-strong)',
				},
			},
			fontFamily: {
				sans: ['Work Sans', 'system-ui', 'sans-serif'],
				mono: ['Space Mono', 'ui-monospace', 'monospace'],
			},
			borderRadius: { lg: 'var(--radius-sm)', xl: 'var(--radius-sm)', '2xl': 'var(--radius-sm)' },
		},
	},
};

module.exports = {
	content: ['./app/pages/*.html', './app/styles/site.js'],
	// Built at runtime in natural_language_processing.html, so the scanner can't see them
	safelist: ['accent-indigo-500', 'accent-rose-500'],
	...theme,
};
