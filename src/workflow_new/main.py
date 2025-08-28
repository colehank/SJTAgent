# %%
from lmitf import TemplateLLM
import os.path as op
import json
import asyncio
from tqdm import tqdm


class SJTAgent:
    def __init__(
        self, 
        situation_theme="大学校园里的日常生活", 
        max_concurrency: int = 100
        ):
        """
        situation_theme: 场景主题
        max_concurrency: 最大并发度（根据你的接口限速能力调整）
        """
        self.situation_theme = situation_theme
        self.max_concurrency = max_concurrency

        # Get the directory where this module is located
        current_dir = op.dirname(op.abspath(__file__))
        
        ba_prompt = op.join(current_dir, "prompts", "behavior_adapter.py")
        sb_prompt_a = op.join(current_dir, "prompts", "scenario_builder_a.py")
        sb_prompt_b = op.join(current_dir, "prompts", "scenario_builder_b.py")
        td_prompt = op.join(current_dir, "prompts", "trait_decoder.py")

        self.ba = TemplateLLM(ba_prompt)
        self.sb_a = TemplateLLM(sb_prompt_a)
        self.sb_b = TemplateLLM(sb_prompt_b)
        self.td = TemplateLLM(td_prompt)

    async def _generate_items(
        self, 
        trait_name, 
        item, 
        n_item,
        model = 'gpt-4o',
        ):
        final_item = {}
        final_item['source'] = item

        # 放线程池，避免阻塞事件循环
        res_td = await asyncio.to_thread(
            self.td.call,
            trait_name=trait_name,
            item=item,
            response_format="json",
            model=model
        )

        cues = await asyncio.to_thread(
            self.sb_a.call,
            trait_name=trait_name,
            situation_theme=self.situation_theme,
            n_cue=n_item,
            cognitive=res_td["cognitive"],
            emotional=res_td["emotional"],
            behavioral=res_td["behavioral"],
            response_format="json",
            model=model
        )
        self.res_td = res_td
        self.res_sb_a = cues

        cue_list = cues.get("cues", [])
        sem = asyncio.Semaphore(self.max_concurrency)

        async def process_cue(cue):
            async with sem:
                try:
                    res_sb_b = await asyncio.to_thread(
                        self.sb_b.call,
                        trait_name=trait_name,
                        cue=cue,
                        n_situ=1,
                        cognitive=res_td["cognitive"],
                        emotional=res_td["emotional"],
                        behavioral=res_td["behavioral"],
                        response_format="json",
                        model=model
                    )

                    res_ba = await asyncio.to_thread(
                        self.ba.call,
                        situation=res_sb_b["situation"][0],
                        trait_name=trait_name,
                        cognitive=res_td["cognitive"],
                        emotional=res_td["emotional"],
                        behavioral=res_td["behavioral"],
                        response_format="json",
                        model=model
                    )

                    return {
                        "situation": res_sb_b["situation"][0],
                        "options": res_ba["options"],
                    }
                except Exception as e:
                    return {
                        "error": repr(e),
                        "cue": cue,
                    }

        tasks = [process_cue(cue) for cue in cue_list]

        results = []
        for fut in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"Generating {trait_name}'s {n_item} sjts from source item",
        ):
            results.append(await fut)

        final_item['n_item'] = n_item
        final_item['trait_decoder'] = res_td
        final_item['cues'] = cue_list
        if n_item == 1:
            final_item['items'] = results[0]
        else:
            final_item['items'] = results

        self.generated_items = final_item

        return final_item

    def generate_items(self, trait_name, item, n_item, model = 'gpt-4o'):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            result = asyncio.run(self._generate_items(trait_name, item, n_item, model=model))
        else:
            import nest_asyncio

            nest_asyncio.apply()
            result = loop.run_until_complete(self._generate_items(trait_name, item, n_item, model=model))
        return result
    
    def _repr_html_(self):
        if not hasattr(self, 'res_td') or not hasattr(self, 'res_sb_a'):
            return "<p>No generation results available. Please run <code>generate_items</code> first.</p>"
        import pandas as pd
        # Create DataFrame for trait decoder results
        td_df = pd.DataFrame([self.res_td]).T
        td_df.columns = [f'特质解析结果']
        td_df.index.name = '维度'

        # Create DataFrame for scenario builder A results (cues)
        cues_data = self.res_sb_a.get("cues", [])
        sb_a_df = pd.DataFrame(cues_data, columns=['情境线索'])
        sb_a_df.index.name = '序号'
        sb_a_df.index = sb_a_df.index + 1

        # Combine both DataFrames into HTML
        html_output = f"""
        <h3>特质解析结果</h3>
        {td_df.to_html(escape=False)}
        <h3>情境线索生成结果 (共{len(cues_data)}个)</h3>
        {sb_a_df.to_html(escape=False)}
        """

        return html_output
# %%
if __name__ == "__main__":
    ipip120_fp = op.abspath(op.join("..", "datasets", "IPIP", "ipip120_zh.json"))
    traits_fp = op.abspath(op.join("..", "datasets", "IPIP", "meta.json"))

    with open(ipip120_fp, "r", encoding="utf-8") as f:
        ipip120 = json.load(f)
    with open(traits_fp, "r", encoding="utf-8") as f:
        traits = json.load(f)

    generator = SJTAgent(
        situation_theme="大学校园里的日常生活",
        max_concurrency=22
    )

    facet_id = "N1"
    trait_name = traits[facet_id]["facet_name"]
    item = ipip120[facet_id]['items']["1"]["item"]
    n_item = 2  # 你当前的参数

    items = generator.generate_items(trait_name, item, n_item)
# %%
