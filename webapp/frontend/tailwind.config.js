/** Monochrome palette derived from the subject (see spec section 5). */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#141414',
        paper: '#FCFCFC',
        slate: '#6B6B6B',
        grid: '#B8B8B8',
        divider: '#E5E5E5',
        fill: '#ECECEC',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        serif: ['Georgia', 'serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
    },
  },
  plugins: [],
}
