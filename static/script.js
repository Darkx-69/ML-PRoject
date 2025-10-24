// static/script.js

// --- Get Data Embedded in HTML ---
const clusterDescriptionsData = JSON.parse(document.getElementById('cluster-descriptions-data').textContent);
const qualityDescriptionsData = JSON.parse(document.getElementById('quality-descriptions-data').textContent);

// --- Get DOM Elements ---
const simpleForm = document.getElementById('simple-form');
const advancedForm = document.getElementById('advanced-form');
const toggleSimpleBtn = document.getElementById('toggle-simple');
const toggleAdvancedBtn = document.getElementById('toggle-advanced');

const loadingDiv = document.getElementById('loading');
const simpleResultsDiv = document.getElementById('simple-prediction-output');
const advancedResultsDiv = document.getElementById('advanced-prediction-output');

// Description placeholders and selects for Simple Form
const clusterDescEl = document.getElementById('cluster-description');
const qualityDescEl = document.getElementById('quality-description');
const clusterSelect = document.getElementById('Cluster');
const qualitySelect = document.getElementById('OverallQual');

// --- Toggle Logic ---
function showSimpleForm() {
    simpleForm.classList.remove('hidden');
    advancedForm.classList.add('hidden');
    // Simple button active styles
    toggleSimpleBtn.classList.remove('bg-transparent', 'text-slate-700');
    toggleSimpleBtn.classList.add('bg-indigo-600', 'text-white', 'shadow');
    // Advanced button inactive styles
    toggleAdvancedBtn.classList.remove('bg-green-600', 'text-white', 'shadow');
    toggleAdvancedBtn.classList.add('bg-transparent', 'text-slate-700');

    simpleResultsDiv.classList.add('hidden');
    advancedResultsDiv.classList.add('hidden');
}

function showAdvancedForm() {
    simpleForm.classList.add('hidden');
    advancedForm.classList.remove('hidden');
    // Simple button inactive styles
    toggleSimpleBtn.classList.remove('bg-indigo-600', 'text-white', 'shadow');
    toggleSimpleBtn.classList.add('bg-transparent', 'text-slate-700');
    // Advanced button active styles
    toggleAdvancedBtn.classList.remove('bg-transparent', 'text-slate-700');
    toggleAdvancedBtn.classList.add('bg-green-600', 'text-white', 'shadow');

    simpleResultsDiv.classList.add('hidden');
    advancedResultsDiv.classList.add('hidden');
}

toggleSimpleBtn.addEventListener('click', showSimpleForm);
toggleAdvancedBtn.addEventListener('click', showAdvancedForm);

// --- Description Update Logic ---
function updateClusterDesc() {
    const selectedCluster = clusterSelect.value;
    clusterDescEl.textContent = clusterDescriptionsData[selectedCluster] || '';
    clusterDescEl.classList.toggle('hidden', !clusterDescEl.textContent); // Hide if empty
}

function updateQualityDesc() {
    const selectedQuality = qualitySelect.value;
    qualityDescEl.textContent = qualityDescriptionsData[selectedQuality] || '';
    qualityDescEl.classList.toggle('hidden', !qualityDescEl.textContent); // Hide if empty
}

// Add event listeners for dropdown changes
clusterSelect.addEventListener('change', updateClusterDesc);
qualitySelect.addEventListener('change', updateQualityDesc);

// Run on page load to set initial descriptions
updateClusterDesc();
updateQualityDesc();

// --- Simple Form Submission ---
simpleForm.addEventListener('submit', async function(event) {
    event.preventDefault();
    loadingDiv.classList.remove('hidden');
    simpleResultsDiv.classList.add('hidden');
    advancedResultsDiv.classList.add('hidden');

    const form = event.target;
    const data = {
        Neighborhood: form.Neighborhood.value,
        Cluster: form.Cluster.value,
        LotArea: form.LotArea.value,
        OverallQual: parseInt(form.OverallQual.value),
        run_price_comparison: form['price-comparison-toggle'].checked
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();

        loadingDiv.classList.add('hidden');
        simpleResultsDiv.classList.remove('hidden');

        // Populate standard results
        document.getElementById('s-predicted-price').textContent = result.predicted_price || 'N/A';
        document.getElementById('s-lot-area-used').textContent = result.lot_area_used || 'N/A';
        document.getElementById('s-avg-price').textContent = result.avg_price || 'N/A';

        // Populate similar house details
        const detailsDiv = document.getElementById('s-actual-house-details');
        const naP = document.getElementById('s-actual-house-na');
        if (result.actual_house && result.actual_house.found) {
            document.getElementById('s-actual-price').textContent = result.actual_house.price || 'N/A';
            document.getElementById('s-actual-neighborhood').textContent = result.actual_house.neighborhood || 'N/A';
            document.getElementById('s-actual-cluster').textContent = result.actual_house.cluster || 'N/A';
            document.getElementById('s-actual-quality').textContent = result.actual_house.quality || 'N/A';
            document.getElementById('s-actual-lot-area').textContent = result.actual_house.lot_area || 'N/A';
            detailsDiv.classList.remove('hidden');
            naP.classList.add('hidden');
        } else {
            detailsDiv.classList.add('hidden');
            naP.classList.remove('hidden');
        }

        // Populate size comparison
        const sizeDetailsDiv = document.getElementById('s-size-comparison-details');
        const sizeNaP = document.getElementById('s-size-comparison-na');
        if (result.size_comparison && result.size_comparison.found) {
            document.getElementById('s-size-target-lot').textContent = result.size_comparison.target_lot_area || 'N/A';
            document.getElementById('s-size-highest-price').textContent = result.size_comparison.highest_price || 'N/A';
            document.getElementById('s-size-highest-neighborhood').textContent = result.size_comparison.highest_neighborhood || 'N/A';
            document.getElementById('s-size-lowest-price').textContent = result.size_comparison.lowest_price || 'N/A';
            document.getElementById('s-size-lowest-neighborhood').textContent = result.size_comparison.lowest_neighborhood || 'N/A';
            sizeDetailsDiv.classList.remove('hidden');
            sizeNaP.classList.add('hidden');
        } else {
            sizeDetailsDiv.classList.add('hidden');
            sizeNaP.classList.remove('hidden');
        }

        // Populate price comparison
        const priceSection = document.getElementById('s-price-comparison-section');
        const priceDetailsDiv = document.getElementById('s-price-comparison-details');
        const priceNaP = document.getElementById('s-price-comparison-na');
        const priceTableBody = document.getElementById('s-price-comparison-table-body');
        priceSection.classList.add('hidden');
        priceDetailsDiv.classList.add('hidden');
        priceNaP.classList.add('hidden');
        priceTableBody.innerHTML = '';

        if (result.price_comparison && result.price_comparison.run) {
            priceSection.classList.remove('hidden');
            if (result.price_comparison.found && result.price_comparison.neighborhoods) {
                document.getElementById('s-price-target').textContent = result.price_comparison.target_price || 'N/A';
                result.price_comparison.neighborhoods.forEach(hood => {
                    const row = `
                        <tr class="text-slate-700 text-sm hover:bg-slate-50">
                            <td class="px-3 py-2 whitespace-nowrap">${hood.neighborhood || 'N/A'}</td>
                            <td class="px-3 py-2 whitespace-nowrap">${hood.quality || 'N/A'}</td>
                            <td class="px-3 py-2 whitespace-nowrap">${hood.liv_area || 'N/A'}</td>
                            <td class="px-3 py-2 whitespace-nowrap">${hood.lot_area || 'N/A'}</td>
                        </tr>
                    `;
                    priceTableBody.innerHTML += row;
                });
                priceDetailsDiv.classList.remove('hidden');
            } else {
                priceNaP.classList.remove('hidden');
            }
        }
    } catch (error) {
        console.error('Error fetching simple prediction:', error);
        loadingDiv.classList.add('hidden');
        simpleResultsDiv.classList.remove('hidden');
        simpleResultsDiv.innerHTML = '<p class="text-red-600 font-medium p-4 text-center">⚠️ An error occurred while getting the prediction. Please try again.</p>';
    }
});

// --- Advanced Form Submission ---
advancedForm.addEventListener('submit', async function(event) {
    event.preventDefault();
    loadingDiv.classList.remove('hidden');
    simpleResultsDiv.classList.add('hidden');
    advancedResultsDiv.classList.add('hidden');
    const form = event.target;
    const data = {
        Neighborhood: form.Adv_Neighborhood.value,
        OverallQual: parseInt(form.Adv_OverallQual.value),
        LotArea: parseInt(form.Adv_LotArea.value),
        GrLivArea: parseInt(form.Adv_GrLivArea.value),
        TotalBsmtSF: parseInt(form.Adv_TotalBsmtSF.value),
        TotalBath: parseFloat(form.Adv_TotalBath.value),
        AgeSold: parseInt(form.Adv_AgeSold.value)
    };

    try {
        const response = await fetch('/predict_advanced', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        loadingDiv.classList.add('hidden');
        advancedResultsDiv.classList.remove('hidden');
        document.getElementById('a-predicted-price').textContent = result.predicted_price || 'N/A';

    } catch (error) {
        console.error('Error fetching advanced prediction:', error);
        loadingDiv.classList.add('hidden');
        advancedResultsDiv.classList.remove('hidden');
        advancedResultsDiv.innerHTML = '<p class="text-red-600 font-medium p-4 text-center">⚠️ An error occurred while getting the prediction. Please check your inputs and try again.</p>';
    }
});