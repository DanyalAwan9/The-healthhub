// Offline demo content, used only when the API call fails (backend not
// running). No default identity lives here — a signed-out / no-profile state
// always shows the real "create a profile" flow, never a fake person.

export const MOCK_DASHBOARD = {
  calorie_target: 1860, protein_target: 128, tdee: 2210, bmi: 22.7,
  streak_days: 6, weight_change_kg: -1.8,
  activity: [
    { icon: 'activity', text: 'Fitness assessment updated', when: '2h ago' },
    { icon: 'sparkles', text: 'Skin analysis — Combination, 82%', when: 'yesterday' },
    { icon: 'chat', text: 'Asked the nutritionist about roti vs bread', when: '2 days ago' },
    { icon: 'scale', text: 'Logged weight 61.0 kg', when: '3 days ago' },
  ],
};

export const MOCK_FITNESS = {
  bmr: 1410, tdee: 2210, calorie_target: 1860,
  macros: { protein_g: 128, carbs_g: 190, fat_g: 55 },
  meal_plan: {
    days: [
      { day: 'Mon', meals: [['Breakfast', '2 egg omelette + 1 roti + tea'], ['Lunch', 'Chicken karahi (150g) + 1 roti + salad'], ['Snack', 'Greek yoghurt + banana'], ['Dinner', 'Daal chana + 1 roti + cucumber']] },
      { day: 'Tue', meals: [['Breakfast', 'Vegetable poha + tea'], ['Lunch', 'Beef seekh (2) + 1 roti + raita'], ['Snack', 'Roasted chana (40g)'], ['Dinner', 'Palak paneer + 1 roti']] },
      { day: 'Wed', meals: [['Breakfast', 'Anda paratha (1) + tea'], ['Lunch', 'Chicken pulao (1 cup) + salad'], ['Snack', 'Apple + peanut butter'], ['Dinner', 'Fish curry + 1 roti']] },
    ],
    daily_cost_pkr: 233, within_budget: true,
  },
  workout: {
    split: 'Full body · 3 days/week',
    days: [
      { day: 'Day 1', items: ['Goblet squat 3×10', 'Push-up 3×10', '1-arm row 3×10', 'Plank 3×40s'] },
      { day: 'Day 2', items: ['Romanian deadlift 3×10', 'Incline press 3×10', 'Lat pulldown 3×12', 'Dead bug 3×12'] },
      { day: 'Day 3', items: ['Split squat 3×10', 'DB shoulder press 3×10', 'Cable row 3×12', 'Hanging knee raise 3×12'] },
    ],
  },
};

export const MOCK_CHAT_HISTORY = [
  { role: 'bot', content: "Hi! I'm your nutrition assistant. Ask me about calories, Pakistani foods or grocery prices." },
];

export function mockChatReply(message) {
  const q = (message || '').toLowerCase();
  if (q.includes('roti') || q.includes('bread')) {
    return 'One medium roti (~90 kcal, 3 g protein) beats two slices of white bread for fibre and satiety.';
  }
  if (q.includes('hungry')) {
    return 'Constant hunger usually means too few calories, not enough protein, or too little fibre. Add an egg or a cup of yoghurt to each meal.';
  }
  return "Here's a quick take: aim for a slight calorie deficit, keep protein around 1.6 g per kg body-weight, and lean on daal, eggs, chicken and yoghurt for cheap protein.";
}

export const MOCK_SKIN_RESULT = {
  skin_type: 'Combination', confidence: 82,
  reasoning: 'Visible shine across the T-zone with matte, slightly tight cheeks — classic combination pattern.',
  care_level: 'light', conditions: ['Blemishes / breakouts'],
  concerns: [
    { name: 'Blemishes', severity: 'mild' },
    { name: 'Uneven texture', severity: 'mild' },
    { name: 'Redness', severity: 'clear' },
  ],
  routine: [
    { step: 'Gentle cleanser', product: 'CeraVe Foaming Cleanser', price_pkr: 2450, why: 'Cheapest gentle pick for combination skin', where_to_buy: 'Daraz' },
    { step: 'Moisturizer', product: 'Cetaphil Daily Oil-Free Moisturizer', price_pkr: 2100, why: 'Lightweight, non-comedogenic', where_to_buy: 'Pharmacy' },
    { step: 'Sunscreen', product: 'ROTEX Sunblock SPF 60', price_pkr: 950, why: 'Best value broad-spectrum', where_to_buy: 'Daraz' },
  ],
  supplements: [
    { name: 'Abbott Pakistan Surbex-Z', price_pkr: 480, dosage: '1 tablet', timing: 'after breakfast', why: 'Zinc + Vitamin C for skin repair' },
  ],
  supplement_note: "This combination is within safe daily limits. Don't add other zinc / vitamin-C / multivitamin products alongside it.",
  diet: [
    '2.5–3 L water/day',
    'More seasonal fruit, leafy greens, dahi, nuts',
    'Less deep-fried food, sugary chai and bakery items',
  ],
  brands_note: 'Gentle products only — no retinoids or benzoyl peroxide. Supplements are Pakistani brands only.',
};

export const MOCK_PROGRESS_ENTRIES = [
  { date: '2026-08-08', weight_kg: 62.8, waist_cm: 79, chest_cm: 93 },
  { date: '2026-08-15', weight_kg: 62.1, waist_cm: 78, chest_cm: 93 },
  { date: '2026-08-22', weight_kg: 61.6, waist_cm: 77.5, chest_cm: 92.5 },
  { date: '2026-08-29', weight_kg: 61.2, waist_cm: 77, chest_cm: 92.5 },
  { date: '2026-09-05', weight_kg: 61.0, waist_cm: 76.5, chest_cm: 92 },
];
