// Offline reference workbench browser flow; no external model inference.
const {chromium}=require('playwright');
const fs=require('fs');
(async()=>{
  const base=process.argv[2],out=process.argv[3];
  if(!base||!out||fs.existsSync(out))throw Error('Supply local URL and new receipt path');
  const url=new URL(base);if(!['127.0.0.1','localhost'].includes(url.hostname))throw Error('Loopback only');
  const browser=await chromium.launch({headless:true,executablePath:chromium.executablePath(),args:['--no-sandbox']});
  try{
    const page=await browser.newPage({viewport:{width:1280,height:900}}),errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.goto(base);await page.waitForFunction(()=>document.querySelector('#mode-pill').textContent==='Reference replay');
    await page.locator('#question').fill('How do I convert between tuples and lists?');
    await page.locator('#ask-button').click();
    await page.waitForFunction(()=>document.querySelector('#task-status').textContent==='Ready for your review');
    const seed=await page.locator('#work-result').innerText();if(!seed.startsWith('3 scripted calls'))throw Error(seed);
    await page.locator('#approve').click();await page.waitForFunction(()=>document.querySelector('#task-status').textContent==='Approved');
    await page.locator('#repeat').click();await page.waitForFunction(()=>document.querySelector('#cache-result').textContent==='Exact answer reused');
    await page.waitForFunction(()=>document.querySelector('#work-result').textContent.startsWith('0 scripted calls'));
    await page.locator('#approve').click();await page.waitForFunction(()=>document.querySelector('#task-status').textContent==='Approved');
    await page.screenshot({path:'/tmp/documentation-desktop.png',fullPage:true});
    await page.locator('#memory-button').click();await page.locator('#memory-dialog').waitFor({state:'visible'});
    if(await page.locator('#memory-items .scope-card').count()!==1)throw Error('Expected one original memory source');
    page.once('dialog',d=>d.accept());await page.locator('#clear-memory').click();
    await page.waitForFunction(()=>document.querySelector('#memory-message').textContent.includes('context cleared'));
    await page.keyboard.press('Escape');
    await page.setViewportSize({width:390,height:844});
    if(!await page.locator('#memory-button').isVisible())throw Error('Memory control unavailable on mobile');
    const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth+1);
    await page.screenshot({path:'/tmp/documentation-mobile.png',fullPage:true});
    if(overflow||errors.length)throw Error(JSON.stringify({overflow,errors}));
    fs.writeFileSync(out,JSON.stringify({evidence_type:'headless Chromium offline reference UI fixture',
      viewports:[[1280,900],[390,844]],observed_scripted_calls:[3,0],memory_list_and_clear:true,
      mobile_horizontal_overflow:false,javascript_exceptions:errors,live_quality_or_savings_measured:false},null,2)+'\n');
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
