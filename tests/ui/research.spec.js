const {test, expect} = require("@playwright/test");

test("research search, filters, paper evidence, and lineage work without remote requests", async ({page}) => {
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  const remoteRequests = [];
  await page.route("**/*", route => {
    const url = new URL(route.request().url());
    if (url.hostname !== "localhost" && url.hostname !== "127.0.0.1") {
      remoteRequests.push(url.href);
      return route.abort();
    }
    return route.continue();
  });
  await page.goto("/");
  await expect(page.locator("#stats")).toContainText("150");
  await page.getByRole("button", {name: "ASML & EUCLIDES", exact: true}).click();
  await expect(page.locator("#results .card").first()).toContainText("EUCLIDES");
  await expect(page.locator("#notice")).toContainText("not generated answers");
  await page.locator("#filter-country").selectOption("JP");
  await page.getByRole("button", {name: "Apply filters", exact: true}).click();
  await expect(page.locator("#search-button")).toBeEnabled();
  await expect(page.locator("#results .card").first()).toContainText("Japan");
  const countryLabels = await page.locator("#results .card-top").allTextContents();
  expect(countryLabels.every(value => value.includes("Japan"))).toBeTruthy();
  await page.locator("#results .paper-title").first().click();
  await expect(page.locator("#paper-dialog")).toBeVisible();
  await expect(page.locator("#paper-content")).toContainText("Indexed source passages");
  await expect(page.locator("#paper-content")).toContainText("Semantically related research");
  await expect(page.locator("#paper-content")).toContainText("not proof of citation");
  await page.getByRole("button", {name: "Close paper details"}).click();
  await page.getByRole("button", {name: "Explore the lineage", exact: true}).click();
  await expect(page.locator("#timeline .milestone")).toHaveCount(8);
  await expect(page.locator("#history-note")).toContainText("ASML");
  await expect(page.locator("#citation-list .citation-row").first()).toBeVisible();
  await page.locator("#timeline details").first().locator("summary").click();
  await expect(page.locator("#timeline")).toContainText("retrospective");
  await page.screenshot({path: ".local/lineage.png", fullPage: true});
  await page.getByRole("button", {name: "Discover research", exact: true}).click();
  await page.getByRole("button", {name: "Clear", exact: true}).click();
  await page.getByRole("button", {name: "Light sources", exact: true}).click();
  await expect(page.locator("#search-button")).toBeEnabled();
  await expect(page.locator("#results .card").first()).toContainText(/plasma|source/i);
  await page.screenshot({path: ".local/search.png", fullPage: true});
  expect(errors).toEqual([]);
  expect(remoteRequests).toEqual([]);
});

test("invalid dates and empty filter results show explicit feedback", async ({page}) => {
  await page.goto("/");
  await expect(page.locator("#filter-country")).toBeVisible();
  await page.locator("[name=year_from]").fill("2000");
  await page.locator("[name=year_to]").fill("1990");
  await page.getByRole("button", {name: "Apply filters", exact: true}).click();
  await expect(page.locator("#notice")).toContainText("Start year must not exceed end year");
  await page.locator("[name=year_from]").fill("2099");
  await page.locator("[name=year_to]").fill("2100");
  await page.getByRole("button", {name: "Apply filters", exact: true}).click();
  await expect(page.locator("#results")).toContainText("No passages match these filters");
});

test("layout is usable at a narrow viewport", async ({page}) => {
  await page.setViewportSize({width: 390, height: 844});
  await page.goto("/");
  await expect(page.locator("#filter-country")).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
});
