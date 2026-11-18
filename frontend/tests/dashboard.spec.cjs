const { test, expect } = require('@playwright/test');

// API responses are controlled; rendering uses bundled, real Leaflet.
test('contract, switching trips, duplicate prevention, retry and connection failure', async ({page}, testInfo) => {
 await page.route('https://**/*', route => route.abort());
 let calls = 0, fail = false;
 await page.route('http://127.0.0.1:8000/**', async route => {
   const path = new URL(route.request().url()).pathname;
   if(path==='/health') return route.fulfill({json:{status:'ready',trip_count:2}});
   if(path==='/trips') return route.fulfill({json:{trip_ids:['0001','0099']}});
   calls++;
   expect(route.request().method()).toBe('POST');
   expect(route.request().postData()).toBeNull();
   await new Promise(r=>setTimeout(r,250));
   if(fail) return route.fulfill({status:500,json:{detail:'Trip data or scoring is unavailable'}});
   const id = path.split('/')[2], second = id==='0099', x=second?80:77;
   return route.fulfill({json:{trip_id:id,observation_count:1,observations:[[x,12,123]],
    predicted_route:[{arc_id:1,coordinates:[[x,12],[x+.01,12.01]]},{arc_id:7,coordinates:[[x+.1,12.1],[x+.11,12.11]]}],
    metrics:{pointwise_accuracy:second?.5:.9,processing_seconds:second?2:1}}});
 });
 await page.goto('/');
 await expect(page.locator('#trip')).toHaveValue('0001');
 await page.locator('#run').click();
 await expect(page.locator('#run')).toBeDisabled();
 await expect(page.locator('#accuracy')).toHaveText('90.0%');
 expect(calls).toBe(1);
 await page.screenshot({path:testInfo.outputPath('dark-desktop.png'),fullPage:true});
 expect(await page.evaluate(async ()=>(await import('/src/main.js')).routes.getLayers().length)).toBe(2);
 expect(await page.evaluate(async ()=>{ const p=(await import('/src/main.js')).gps.getLayers()[0].getLatLng(); return [p.lat,p.lng]; })).toEqual([12,77]);
 await page.locator('#trip').selectOption('0099');
 await expect(page.locator('#accuracy')).toHaveText('—');
 expect(await page.evaluate(async ()=>(await import('/src/main.js')).routes.getLayers().length)).toBe(0);
 await page.locator('#run').click();
 await expect(page.locator('#accuracy')).toHaveText('50.0%');
 await expect(page.locator('#seconds')).toHaveText('2.000 s');
 await expect(page.locator('#count')).toHaveText('1');
 expect(await page.evaluate(async ()=>{ const m=await import('/src/main.js'); return m.map.getBounds().contains([12,80]) && !m.map.getBounds().contains([12,77]); })).toBe(true);
 await page.locator('#route-toggle').uncheck();
 expect(await page.evaluate(async ()=>{const m=await import('/src/main.js');return m.map.hasLayer(m.routes)})).toBe(false);
 await page.locator('#route-toggle').check();
 expect(await page.evaluate(async ()=>{const m=await import('/src/main.js');return m.map.hasLayer(m.routes)})).toBe(true);
 expect(await page.evaluate(async ()=>(await import('/src/main.js')).routes.getLayers().map(l=>l.getLatLngs().length))).toEqual([2,2]);
 await page.locator('#gps-toggle').uncheck();
 expect(await page.evaluate(async ()=>{const m=await import('/src/main.js');return m.map.hasLayer(m.gps)})).toBe(false);
 await page.locator('#gps-toggle').check();
 fail=true;
 await page.locator('#run').click();
 await expect(page.locator('#status')).toContainText('retry');
 await expect(page.locator('#accuracy')).toHaveText('—');
 await expect(page.locator('#run')).toBeEnabled();
 fail=false; await page.locator('#run').click();
 await expect(page.locator('#accuracy')).toHaveText('50.0%');
 await page.locator('#base').fill('http://127.0.0.1:8999');
 await expect(page.locator('#run')).toBeDisabled();
 await expect(page.locator('#accuracy')).toHaveText('—');
 await page.locator('#connect').click();
 await expect(page.locator('#connection-note')).toContainText('Cannot reach API');
 await expect(page.locator('#run')).toBeDisabled();
});

for (const [name, payload, message] of [
 ['wrong trip', {trip_id:'other'}, 'contract'],
 ['invalid geometry', {trip_id:'0001',observation_count:1,observations:[[500,12,123]],predicted_route:[],metrics:{pointwise_accuracy:.9,processing_seconds:1}}, 'Invalid coordinates'],
 ['invalid accuracy', {trip_id:'0001',observation_count:0,observations:[],predicted_route:[],metrics:{pointwise_accuracy:90,processing_seconds:1}}, 'contract'],
]) {
 test('rejects ' + name + ' without displaying partial results', async ({page}, testInfo) => {
   await page.route('https://**/*', r=>r.abort());
   await page.route('http://127.0.0.1:8000/**', r=>{
     const path=new URL(r.request().url()).pathname;
     return r.fulfill({json:path==='/health'?{status:'ready',trip_count:1}:path==='/trips'?{trip_ids:['0001']}:payload});
   });
   await page.goto('/');
   await expect(page.locator('#run')).toBeEnabled();
   await page.locator('#run').click();
   await expect(page.locator('#status')).toContainText(message);
   await expect(page.locator('#accuracy')).toHaveText('—');
   expect(await page.evaluate(async ()=>(await import('/src/main.js')).routes.getLayers().length)).toBe(0);
   await expect(page.locator('#run')).toBeEnabled();
 });
}

test('empty trip lists, invalid JSON, and mobile layout', async ({page}, testInfo) => {
 await page.setViewportSize({width:390,height:844});
 await page.route('https://**/*', r=>r.abort());
 let invalid=false;
 await page.route('http://127.0.0.1:8000/**', r=>{
   if(invalid) return r.fulfill({body:'<html>Wrong service</html>',contentType:'text/html'});
   return r.fulfill({json:r.request().url().endsWith('/health')?{status:'ready',trip_count:0}:{trip_ids:[]}});
 });
 await page.goto('/');
 await expect(page.locator('#status')).toContainText('no trips');
 await expect(page.locator('#run')).toBeDisabled();
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
 invalid=true;
 await page.locator('#connect').click();
 await expect(page.locator('#connection-note')).toContainText('invalid JSON');
 await expect(page.locator('#run')).toBeDisabled();
 await page.screenshot({path:testInfo.outputPath('dark-mobile.png'),fullPage:true});
});
