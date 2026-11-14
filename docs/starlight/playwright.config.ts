import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/browser',
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  workers: 2,
  reporter: [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL: 'http://127.0.0.1:4310',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    reducedMotion: 'reduce',
    launchOptions: {
      executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH,
    },
  },
  projects: [
    {
      name: 'desktop-light',
      use: {
        browserName: 'chromium',
        viewport: { width: 1440, height: 1000 },
        colorScheme: 'light',
      },
    },
    {
      name: 'desktop-dark',
      use: {
        browserName: 'chromium',
        viewport: { width: 1440, height: 1000 },
        colorScheme: 'dark',
      },
    },
    {
      name: 'mobile-light',
      use: {
        browserName: 'chromium',
        viewport: { width: 390, height: 844 },
        colorScheme: 'light',
        isMobile: true,
        hasTouch: true,
      },
    },
    {
      name: 'mobile-dark',
      use: {
        browserName: 'chromium',
        viewport: { width: 390, height: 844 },
        colorScheme: 'dark',
        isMobile: true,
        hasTouch: true,
      },
    },
  ],
  webServer: {
    command: 'npm exec -- astro preview --host 127.0.0.1 --port 4310 --ignore-lock',
    url: 'http://127.0.0.1:4310',
    reuseExistingServer: !process.env.CI,
    stdout: 'ignore',
    stderr: 'pipe',
  },
});
